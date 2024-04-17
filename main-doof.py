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

  def __init__(self, weight_name, config, input_size=None, output_size=None, weights=None, bias=None):
    self.config = config
    self.weight_name = weight_name
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

  def load(self):
    self.weights = load_weight_tensor(self.weight_name)

  def forward(self, x):
    # x is of shape (batch_size, sequence_length, hidden_size).
    return x * self.weights.T + self.bias


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
    pass


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

    self.query_projection = LinearLayer(
      f"model.layers.{self.layer_idx}.self_attn.q_proj.weight",
      config,
      input_size=self.config["hidden_size"],
      output_size=self.config["num_attention_heads"] * self.head_dim
    )
    self.key_projection = LinearLayer(
      f"model.layers.{self.layer_idx}.self_attn.k_proj.weight",
      config,
      input_size=self.config["hidden_size"],
      output_size=self.config["num_key_value_heads"] * self.head_dim
    )
    self.value_projection = LinearLayer(
      f"model.layers.{self.layer_idx}.self_attn.v_proj.weight",
      config,
      input_size=self.config["hidden_size"],
      output_size=self.config["num_key_value_heads"] * self.head_dim
    )
    self.output_projection = LinearLayer(
      f"model.layers.{self.layer_idx}.self_attn.o_proj.weight",
      config,
      input_size=self.config["hidden_size"],
      output_size=self.config["hidden_size"]
    )

    self.rotation_embedding = LlamaRotationEmbedding(self.head_dim, config)

  def load(self):
    # Load weights from the model.
    self.query_projection.load()
    self.key_projection.load()
    self.value_projection.load()
    self.output_projection.load()
  
  def forward(self, x, position_ids=None, attention_mask=None):
    # x is of shape (batch_size, sequence_length, hidden_size).
    # position_ids is of shape (batch_size, sequence_length).
    # attention_mask is of shape (batch_size, sequence_length, sequence_length).
    batch_size, sequence_length, _ = x.shape


class MLP:
  """
  Models a Multi-Layer Perceptron (MLP).
  Uses SwiGLU activation function.
  """

  def __init__(self, layer_idx, config):
    self.config = config
    self.layer_idx = layer_idx
    self.hidden_size = config["hidden_size"]
    self.intermediate_size = config["intermediate_size"]

    self.gate_proj = LinearLayer(
      f"model.layers.{self.layer_idx}.mlp.gate_proj.weight",
      config,
      input_size=self.hidden_size,
      output_size=self.intermediate_size
    )
    self.up_proj = LinearLayer(
      f"model.layers.{self.layer_idx}.mlp.up_proj.weight",
      config,
      input_size=self.hidden_size,
      output_size=self.intermediate_size
    )
    self.down_proj = LinearLayer(
      f"model.layers.{self.layer_idx}.mlp.down_proj.weight",
      config,
      input_size=self.intermediate_size,
      output_size=self.hidden_size
    )

    # SiLU a.k.a. Swish activation function.
    self.activation = torch.nn.SiLU()

  def load(self):
    self.gate_proj.load()
    self.up_proj.load()
    self.down_proj.load()

  def forward(self, x):
    # x is of shape (batch_size, sequence_length, hidden_size).

    # SwiGLU Feed Forward Network = (Swish(x * gate) * up) * down
    return self.down_proj.forward(
      self.activation(self.gate_proj.forward(x)) * self.up_proj.forward(x)
    )


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
    # x is of shape (batch_size, sequence_length, hidden_size).
    variance = x * torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.variance_epsilon)
    return variance * self.weights


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
    
    residual = x

    x = self.input_norm.forward(x)
    x = self.attention.forward(x, position_ids, attention_mask)

    x = residual + x

    residual = x

    self.post_attention_norm.forward(x)
    self.mlp.forward(x)

    x = residual + x

    return (x,)


class LlamaEmbedding:
  """
  Map token ids to embeddings using a lookup table.
  """

  def __init__(self, config):
    self.config = config

  def load(self):
    self.lookup_table = load_weight_tensor("model.embed_tokens.weight")
  
  def forward(self, input_ids):
    pass

class LlamaModel:

  def __init__(self, config):
    self.config = config

    self.embed_tokens = LlamaEmbedding(config)
    self.decoder_layers = [LlamaDecoderLayer(idx, config) for idx in range(config["num_hidden_layers"])]
    self.post_decoder_norm = RMSNorm("model.norm.weight", config)
    self.score_layer = LinearLayer("lm_head.weight", config, input_size=config["hidden_size"], output_size=config["vocab_size"])

    # Create the causal mask.
    # False,True,True...True
    # False,False,True...True
    # False,False,False,True...True
    # ...
    # False,False,False...False
    # Each token can only attend to the tokens before it.
    self.causal_mask = torch.full((config["max_position_embeddings"], config["max_position_embeddings"]), fill_value=True, dtype=torch.bool)
    self.causal_mask = torch.triu(self.causal_mask, diagonal=1)

  def load(self):
    self.embed_tokens.load()

    for layer in self.decoder_layers:
      layer.load()

    self.post_decoder_norm.load()
    self.score_layer.load()
  
  def forward(self, input_ids, position_ids=None, attention_mask=None):
    # input_ids is of shape (batch_size, sequence_length).
    # position_ids is of shape (batch_size, sequence_length).
    # attention_mask is of shape (batch_size, sequence_length).
    pass


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
  # Position ids:    [0, 1, 2, 3, 4, ..., 1, 1, 1, 1]
  # Attention mask:  [1, 1, 1, 1, 1, ..., 0, 0, 0, 0]
  position_ids = torch.arange(max_sequence_length)
  position_ids.masked_fill_(attention_mask[0] == 0, 1)

  # Expand the position ids to match the batch size: (sequence_length) -> (batch_size, sequence_length)
  # Batch size is 1 in this case, so we just add a dimension at the beginning.
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
