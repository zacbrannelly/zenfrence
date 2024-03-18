import logging
import json
import torch
import math
import fire
from safetensors import safe_open
from transformers import LlamaTokenizer

# Set the logging level to debug.
logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)

# Disable autograd, we only want inference.
torch.set_grad_enabled(False)

# From: meta-llama/Llama-2-7b-chat-hf
model_config = {
  "bos_token_id": 1,
  "eos_token_id": 2,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 11008,
  "max_position_embeddings": 4096,
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  # Apparently if "num_key_value_heads" = "num_attention_heads" then it won't use Group Query Attention (GQA), so maybe can skip that implementation for now?
  # Looks like the 70b model uses a different number for this, so we'll need to implement it eventually.
  "num_key_value_heads": 32,
  "rms_norm_eps": 1e-05,
  "vocab_size": 32000
}


with open("weight_index.json", "r") as f:
  weight_index = json.load(f)
  weight_index = weight_index["weight_map"]

MODEL_WEIGHTS = {
  "model-00001-of-00002.safetensors": safe_open("model-00001-of-00002.safetensors", framework='pt', device='cpu'),
  "model-00002-of-00002.safetensors": safe_open("model-00002-of-00002.safetensors", framework='pt', device='cpu'),
}

def load_weight_tensor(weight_name):
  return MODEL_WEIGHTS[weight_index[weight_name]].get_tensor(weight_name)


class LinearLayer:
  """
  Models: y = xW^T + b
  """

  def __init__(self, config, input_size=None, output_size=None, weights=None, bias=None):
    self.config = config
    self.input_size = input_size
    self.output_size = output_size

    # Must be of shape (output_size, input_size).
    self.weights = weights
    if not weights:
      self.weights = torch.zeros(output_size, input_size)

    # Must be of shape (output_size,).
    self.bias = bias
    if not bias:
      self.bias = torch.zeros(output_size)

  def forward(self, x):
    # TODO: Validate that the input size matches the weights.
    # Move input tensor x to MPS device
    x_mps = x.to('mps')
    weights_mps = self.weights.to('mps')
    bias_mps = self.bias.to('mps')

    # Perform matrix multiplication and addition on MPS device
    result = torch.matmul(x_mps, weights_mps.T) + bias_mps

    # Optionally, move result back to CPU if needed
    result = result.to('cpu')

    return result


class LlamaRotationEmbedding:
  """
  Models Rotational Positional Embeddings (RoPE).
  Paper: https://arxiv.org/pdf/2104.09864.pdf
  """

  def __init__(self, head_dim, config):
    self.head_dim = head_dim
    self.config = config

    # inv_freq maps to the /theta symbol in the paper (Section 3.3)
    # Result is of shape (hidden_size / 2,).
    # This calculates the theta value for half of the hidden size.
    self.inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))

  def forward(self, position_ids, query, key):
    # The shape of `position_ids` is:  (batch_size, sequence_length).
    # The shape of `query` and `key` is: (batch_size, num_attention_heads, sequence_length, head_dim).
    batch_size = position_ids.shape[0]

    # (head_dim / 2,) -> (1, head_dim / 2, 1)
    inv_freq_expanded = self.inv_freq[None, :, None].float()

    # (1, head_dim / 2, 1) -> (batch_size, head_dim / 2, 1)
    inv_freq_expanded = inv_freq_expanded.expand(batch_size, -1, 1)

    # (batch_size, sequence_length,) -> (batch_size, 1, sequence_length)
    position_ids_expanded = position_ids[:, None, :]

    # This maps to \theta * m in the paper (Section 3.4.2)
    # Result: (batch_size, head_dim / 2, 1) @ (batch_size, 1, sequence_length) -> (batch_size, head_dim / 2, sequence_length)
    frequencies = inv_freq_expanded.float() @ position_ids_expanded.float()

    # (batch_size, head_dim / 2, sequence_length) -> (batch_size, sequence_length, head_dim / 2)
    frequencies = frequencies.transpose(1, 2)

    # Result: (batch_size, sequence_length, head_dim / 2) -> (batch_size, sequence_length, head_dim)
    emb = torch.cat((frequencies, frequencies), dim=-1)

    # Apply the sine and cosine functions to each of the values.
    # Doesn't change the shape.
    cos = emb.cos()
    sin = emb.sin()

    # (batch_size, sequence_length, head_dim) -> (batch_size, 1, sequence_length, head_dim)
    # I believe this is to help make the next step broadcast correctly.
    # TODO: Broadcasting is dumb because it's math magic, I need to do this in another language to grok it.
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    # TODO: Figure out how in the Pythonic hell this works.
    def rotate_half(x):
      """Rotates half the hidden dims of the input."""
      x1 = x[..., : x.shape[-1] // 2]
      x2 = x[..., x.shape[-1] // 2 :]
      return torch.cat((-x2, x1), dim=-1)

    # See 3.4.2 in the paper for what this is doing conceptually.
    query_rotated = rotate_half(query)
    key_rotated = rotate_half(key)

    query_embed = query * cos + query_rotated * sin
    key_embed = key * cos + key_rotated * sin

    return query_embed, key_embed


class LlamaAttention:
  """
  Models multi-head attention.
  This contains all the attention heads for a single layer.
  """

  def __init__(self, layer_idx, config):
    self.layer_idx = layer_idx
    self.config = config

    # Dimension of a single head.
    # e.g. 128 for a hidden size of 4096 and 32 attention heads.
    self.head_dim = self.config["hidden_size"] // self.config["num_attention_heads"]
    self.num_key_value_groups = self.config["num_attention_heads"] // self.config["num_key_value_heads"]

    self.query_projection = LinearLayer(config, input_size=self.config["hidden_size"], output_size=self.config["num_attention_heads"] * self.head_dim)
    self.key_projection = LinearLayer(config, input_size=self.config["hidden_size"], output_size=self.config["num_key_value_heads"] * self.head_dim)
    self.value_projection = LinearLayer(config, input_size=self.config["hidden_size"], output_size=self.config["num_key_value_heads"] * self.head_dim)
    self.output_projection = LinearLayer(config, input_size=self.config["hidden_size"], output_size=self.config["hidden_size"])

    self.rotation_embedding = LlamaRotationEmbedding(self.head_dim, config)

  def load(self):
    # Load weights from the model.
    self.query_projection.weights = load_weight_tensor(f"model.layers.{self.layer_idx}.self_attn.q_proj.weight")
    self.key_projection.weights = load_weight_tensor(f"model.layers.{self.layer_idx}.self_attn.k_proj.weight")
    self.value_projection.weights = load_weight_tensor(f"model.layers.{self.layer_idx}.self_attn.v_proj.weight")
    self.output_projection.weights = load_weight_tensor(f"model.layers.{self.layer_idx}.self_attn.o_proj.weight")
  
  def forward(self, x, position_ids=None, attention_mask=None):
    # x is of shape (batch_size, sequence_length, hidden_size).
    # position_ids is of shape (batch_size, sequence_length).
    # attention_mask is of shape (batch_size, sequence_length, sequence_length).
    batch_size, sequence_length, _ = x.shape

    # Calculate the query, key, and value vectors.
    query = self.query_projection.forward(x)
    key = self.key_projection.forward(x)
    value = self.value_projection.forward(x)

    # Go from (batch_size, sequence_length, hidden_size) to (batch_size, sequence_length, num_attention_heads, head_dim)
    query = query.view(batch_size, sequence_length, self.config["num_attention_heads"], self.head_dim)

    # Go from (batch_size, sequence_length, hidden_size) to (batch_size, sequence_length, num_key_value_heads, head_dim)
    key = key.view(batch_size, sequence_length, self.config["num_key_value_heads"], self.head_dim)
    value = value.view(batch_size, sequence_length, self.config["num_key_value_heads"], self.head_dim)

    # Go from (batch_size, sequence_length, num_attention_heads, head_dim) to (batch_size, num_attention_heads, sequence_length, head_dim)
    query = query.transpose(1, 2)

    # Go from (batch_size, sequence_length, num_key_value_heads, head_dim) to (batch_size, num_key_value_heads, sequence_length, head_dim)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    # Apple RoPe (Rotational Positional Embeddings).
    # This adds positional information to the query and key vectors.
    query, key = self.rotation_embedding.forward(position_ids, query, key)

    # Repeat the query and key vectors to match the number of attention heads when using GQA.
    if self.num_key_value_groups > 1:
      key = key[:, :, None, :, :].expand(-1, -1, self.num_key_value_groups, -1, -1)
      key = key.reshape(batch_size, self.config["num_attention_heads"], sequence_length, self.head_dim)

      value = value[:, :, None, :, :].expand(-1, -1, self.num_key_value_groups, -1, -1)
      value = value.reshape(batch_size, self.config["num_attention_heads"], sequence_length, self.head_dim)

    # Calculate the Scaled Dot-Product Attention across all the heads in parallel.
    # After this step, attention is of shape (batch_size, num_attention_heads, sequence_length, sequence_length).
    attention = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(self.head_dim)

    # Apply the attention mask.
    if attention_mask is not None:
      attention = attention + attention_mask

    # Apply the softmax function to the attention scores.
    attention = torch.nn.functional.softmax(attention, dim=-1)

    # Apply the attention scores to the value vectors.
    # (batch_size, num_attention_heads, sequence_length, sequence_length) to (batch_size, num_attention_heads, sequence_length, head_dim)
    attention = torch.matmul(attention, value)

    # (batch_size, num_attention_heads, sequence_length, head_dim) to (batch_size, sequence_length, num_attention_heads, head_dim)
    attention = attention.transpose(1, 2).contiguous()

    # Reshape the attention to be the same shape as the input.
    # (batch_size, sequence_length, num_attention_heads, head_dim) to (batch_size, sequence_length, hidden_size)
    attention = attention.reshape(batch_size, sequence_length, self.config["hidden_size"])

    # Apply the output projection.
    return self.output_projection.forward(attention)


class MLP:
  """
  Models a Multi-Layer Perceptron (MLP).
  """

  def __init__(self, layer_idx, config):
    self.config = config
    self.layer_idx = layer_idx
    self.hidden_size = config["hidden_size"]
    self.intermediate_size = config["intermediate_size"]
    self.gate_proj = LinearLayer(config, input_size=self.hidden_size, output_size=self.intermediate_size)
    self.up_proj = LinearLayer(config, input_size=self.hidden_size, output_size=self.intermediate_size)
    self.down_proj = LinearLayer(config, input_size=self.intermediate_size, output_size=self.hidden_size)

    # TODO: The Llama paper says use SwiGLU, but the code uses SiLU.
    # TODO: Implement this part so I can understand what is happening.
    self.activation = torch.nn.SiLU()

  def load(self):
    self.gate_proj.weights = load_weight_tensor(f"model.layers.{self.layer_idx}.mlp.gate_proj.weight")
    self.up_proj.weights = load_weight_tensor(f"model.layers.{self.layer_idx}.mlp.up_proj.weight")
    self.down_proj.weights = load_weight_tensor(f"model.layers.{self.layer_idx}.mlp.down_proj.weight")

  def forward(self, x):
    # Apply the gate projection.
    gate = self.gate_proj.forward(x)
    gate = self.activation(gate)

    # Apply the up projection.
    up = gate * self.up_proj.forward(x)

    # Apply the down projection.
    return self.down_proj.forward(up)


class RMSNorm:
  """
  RMSNorm
  Models:
    rms[i] = sqrt(sum(x[i] ** 2) / hidden_size)
    a[i] = x[i] / rms[i]
  """
  def __init__(self, weight_name, config):
    self.config = config
    self.weight_name = weight_name
    self.variance_epsilon = config["rms_norm_eps"]
  
  def load(self):
    self.weights = load_weight_tensor(self.weight_name)

  def forward(self, x):
    # So apparently x[i] maps to a[i] in the paper, no need to apply the weights to the input.
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    
    # torch.rqrt is the reciprocal square root function (1 / sqrt(x)).
    # The variance_epsilon is added to the denominator to avoid division by zero.
    x = x * torch.rsqrt(variance + self.variance_epsilon)

    # I think self.weights maps to g(i) in the paper!
    # So bloody confusing.
    return self.weights * x


class LlamaDecoderLayer:

  def __init__(self, layer_idx, config):
    self.config = config
    self.layer_idx = layer_idx
    self.input_norm = RMSNorm(
      f"model.layers.{layer_idx}.input_layernorm.weight",
      config,
    )
    self.attention = LlamaAttention(layer_idx, config)
    self.post_attention_norm = RMSNorm(
      f"model.layers.{layer_idx}.post_attention_layernorm.weight",
      config,
    )
    self.mlp = MLP(layer_idx, config)

  def load(self):
    self.attention.load()
    self.input_norm.load()
    self.post_attention_norm.load()
    self.mlp.load()

  def forward(self, x, attention_mask, position_ids):
    # x is of shape (batch_size, sequence_length, hidden_size)
    # position_ids is of shape (batch_size, sequence_length)
    # attention_mask is of shape (batch_size, sequence_length, sequence_length)

    # Store a copy of the input for the residual connection.
    residual = x

    # Normalize the input.
    x = self.input_norm.forward(x)

    x = self.attention.forward(
      x=x,
      attention_mask=attention_mask,
      position_ids=position_ids,
    )

    # Add the residual connection.
    x = residual + x

    # Store a copy of the input for the residual connection.
    residual = x

    x = self.post_attention_norm.forward(x)
    x = self.mlp.forward(x)

    # Add the residual connection.
    x = residual + x

    # TODO: Figure out what output_attentions is for in the original code.
    outputs = (x,)

    return outputs

class LlamaEmbedding:
  """
  Map token ids to embeddings using a lookup table.
  """

  def __init__(self, config):
    self.config = config

  def load(self):
    self.lookup_table = load_weight_tensor("model.embed_tokens.weight")
  
  def forward(self, input_ids):
    return self.lookup_table[input_ids]

class LlamaModel:

  def __init__(self, config):
    self.config = config

    self.embed_tokens = LlamaEmbedding(config)
    self.decoder_layers = [LlamaDecoderLayer(idx, config) for idx in range(config["num_hidden_layers"])]
    self.post_decoder_norm = RMSNorm("model.norm.weight", config)
    self.score_layer = LinearLayer(config, input_size=config["hidden_size"], output_size=config["vocab_size"])

    # Create the causal mask.
    # False,True,True...True
    # False,False,True...True
    # False,False,False,True...True
    # ...
    # Each token can only attend to the tokens before it.
    self.causal_mask = torch.full((config["max_position_embeddings"], config["max_position_embeddings"]), fill_value=True, dtype=torch.bool)
    self.causal_mask = torch.triu(self.causal_mask, diagonal=1)

  def load(self):
    self.embed_tokens.load()

    for layer in self.decoder_layers:
      layer.load()

    self.post_decoder_norm.load()

    # TODO: Do the loading inside the LinearLayer class.
    self.score_layer.weights = load_weight_tensor("lm_head.weight")
  
  def forward(self, input_ids, position_ids=None, attention_mask=None):
    # input_ids is of shape (batch_size, sequence_length).
    # position_ids is of shape (batch_size, sequence_length).
    # attention_mask is of shape (batch_size, sequence_length).

    # Get the input embeddings.
    # Result is of shape (batch_size, sequence_length, hidden_size).
    x = self.embed_tokens.forward(input_ids)

    batch_size, sequence_length = x.shape[:2]
    
    # Causal mask is of shape (sequence_length, sequence_length).
    # Meaning each token has its own mask, and that mask should only allow it to attend to the tokens before it.
    min_dtype = torch.finfo(x.dtype).min
    causal_mask = self.causal_mask.to(dtype=x.dtype, device=x.device) * min_dtype

    # (sequence_length, sequence_length) -> (1, 1, sequence_length, sequence_length)
    causal_mask = causal_mask[None, None, :, :]

    # (1, 1, sequence_length, sequence_length) -> (batch_size, 1, sequence_length, sequence_length)
    causal_mask = causal_mask.expand(batch_size, 1, -1, -1)

    # Apply the attention mask to the causal mask.
    # min_dtype is used to mask the padding tokens, s
    padding_mask = causal_mask[..., :sequence_length].eq(0.0) * attention_mask[:, None, None, :].eq(0.0)
    causal_mask = causal_mask.clone()

    # TODO: Figure out what in the Pythonic hell is going on here.
    causal_mask[..., :sequence_length] = causal_mask[..., :sequence_length].masked_fill(padding_mask, min_dtype)

    for decoder_layer in self.decoder_layers:
      layer_outputs = decoder_layer.forward(
        x,
        attention_mask=causal_mask,
        position_ids=position_ids,
      )
      x = layer_outputs[0]
      LOG.info(f"Completed layer {decoder_layer.layer_idx}.")
    
    # Normalize the output.
    x = self.post_decoder_norm.forward(x)

    # Apply the score layer to get the logits (probabilities of each token in the vocabulary).
    logits = self.score_layer.forward(x)
    logits = logits.float()

    return (logits,)


def inference(model: LlamaModel, tokenizer: LlamaTokenizer, prompt: str, max_length: int = 100):
  # Tokenize the input.
  # Result will be of shape (1, sequence_length).
  tokenizer.pad_token = tokenizer.eos_token
  tokenizer_result = tokenizer(
    prompt,
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=model.config["max_position_embeddings"]
  )

  input_ids = tokenizer_result["input_ids"]
  attention_mask = tokenizer_result["attention_mask"]

  # Ensure the input sequence is not too long.
  max_sequence_length = model.config["max_position_embeddings"]
  if input_ids.shape[1] > max_sequence_length:
    raise ValueError(f"Input sequence length is too long, must be less than or equal to {max_sequence_length}.")
  
  # Apply the position ids.
  position_ids = torch.arange(max_sequence_length)
  position_ids.masked_fill_(attention_mask[0] == 0, 1)

  # Expand the position ids to match the batch size: (sequence_length) -> (batch_size, sequence_length)
  position_ids = position_ids[None, :]

  # Calculate the current position using the attention mask.
  current_position_id = attention_mask[0].sum() - 1

  # Do greedy decoding.
  for _ in range(max_length):
    # Get the model's logits for the next token.
    outputs = model.forward(
      input_ids,
      position_ids=position_ids,
      attention_mask=attention_mask
    )

    # Shape: (batch_size, sequence_length, vocab_size)
    logits = outputs[0]

    # Get the token id with the highest probability.
    next_token = torch.argmax(logits, dim=-1)[0, current_position_id]

    # Put the next token at the end of the input sequence.
    next_pos = current_position_id + 1
    input_ids[0][next_pos] = next_token

    # Update the attention mask and position ids.
    attention_mask[0][next_pos] = 1
    position_ids[0][next_pos] = position_ids[0][current_position_id] + 1

    # If the next token is the end of sequence token, stop.
    if next_token == tokenizer.eos_token_id:
      break

    current_position_id = next_pos

    LOG.info(f"Current output: {tokenizer.decode(input_ids[0], skip_special_tokens=True)}")

  # Decode the output sequence.
  output = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
  output = output[0][len(prompt):]
  return output


class CLIWrapper:

  def __init__(self):
    LOG.info("Loading Llama model...")
    self.model = LlamaModel(model_config)
    self.model.load()

    LOG.info("Loading Llama tokenizer...")
    # TODO: Implement this myself to understand what is happening here.
    self.tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
  
  def inference(self, prompt):
    return inference(self.model, self.tokenizer, prompt)
  

if __name__ == "__main__":
  fire.Fire(CLIWrapper)
