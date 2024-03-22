import fire
import json
import mlx.core as mx
import math
import numpy as np
import torch
import logging

from transformers import LlamaTokenizer

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)

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
  "num_key_value_heads": 32,
  "rms_norm_eps": 1e-05,
  "vocab_size": 32000
}


with open("weight_index.json", "r") as f:
  weight_index = json.load(f)
  weight_index = weight_index["weight_map"]

MODEL_WEIGHTS = {
  "model-00001-of-00002.safetensors": mx.load("model-00001-of-00002.safetensors"),
  "model-00002-of-00002.safetensors": mx.load("model-00002-of-00002.safetensors"),
}

def load_weight_tensor(weight_name):
  return MODEL_WEIGHTS[weight_index[weight_name]][weight_name]


class LinearLayer:
  def __init__(self, weight_name):
    self.weight_name = weight_name
  
  def load(self):
    self.weights = load_weight_tensor(self.weight_name)

  def forward(self, x):
    return x @ self.weights.T


class LlamaRotationalEmbedding:
  def __init__(self, head_size, config):
    self.head_size = head_size
    self.config = config
    self.inv_freq = 1.0 / (10000 ** (mx.arange(0, self.head_size, 2, dtype=mx.float16) / self.head_size))

  def forward(self, position_ids, query, key):
    batch_size = position_ids.shape[0]

    # Expand to the batch size
    inv_freq_expanded = self.inv_freq[None, :, None]
    inv_freq_expanded = mx.repeat(inv_freq_expanded, repeats=batch_size, axis=0)

    position_ids = position_ids[:, None, :]

    frequencies = inv_freq_expanded @ position_ids

    frequencies = frequencies.transpose(0, 2, 1)

    emb = mx.concatenate([frequencies, frequencies], axis=-1)

    cos = mx.cos(emb)
    sin = mx.sin(emb)

    # Unsqueezing
    cos = cos[:, None, :, :]
    sin = sin[:, None, :, :]

    # TODO: Figure out how in the Pythonic hell this works.
    def rotate_half(x):
      """Rotates half the hidden dims of the input."""
      x1 = x[..., : x.shape[-1] // 2]
      x2 = x[..., x.shape[-1] // 2 :]
      return mx.concatenate((-x2, x1), axis=-1)

    query_rotated = rotate_half(query)
    key_rotated = rotate_half(key)

    query_embed = query * cos + query_rotated * sin
    key_embed = key * cos + key_rotated * sin

    return query_embed, key_embed


class LlamaAttention:
  def __init__(self, layer_idx, config):
    self.layer_idx = layer_idx
    self.config = config
    self.head_size = config["hidden_size"] // config["num_attention_heads"]

    # (hidden_size, num_heads * head_size)
    self.query_proj = LinearLayer(f"model.layers.{layer_idx}.self_attn.q_proj.weight")
    self.output_proj = LinearLayer(f"model.layers.{layer_idx}.self_attn.o_proj.weight")

    # (hidden_size, num_key_value_heads * head_size)
    self.key_proj = LinearLayer(f"model.layers.{layer_idx}.self_attn.k_proj.weight")
    self.value_proj = LinearLayer(f"model.layers.{layer_idx}.self_attn.v_proj.weight")

    self.rotational_embedding = LlamaRotationalEmbedding(self.head_size, config)

  def load(self):
    self.query_proj.load()
    self.key_proj.load()
    self.value_proj.load()
    self.output_proj.load()
  
  def forward(self, x, position_ids, attention_mask):
    # x is (batch_size, seq_len, hidden_size)
    # position_ids is (batch_size, seq_len)
    # attention_mask is (batch_size, seq_len, seq_len)

    batch_size, seq_len, hidden_size = x.shape

    query = self.query_proj.forward(x)
    key = self.key_proj.forward(x)
    value = self.value_proj.forward(x)

    # (batch_size, seq_len, hidden_size) -> (batch_size, seq_len, num_attention_heads, head_size)
    query = query.reshape(batch_size, seq_len, self.config["num_attention_heads"], self.head_size)
    key = key.reshape(batch_size, seq_len, self.config["num_key_value_heads"], self.head_size)
    value = value.reshape(batch_size, seq_len, self.config["num_key_value_heads"], self.head_size)

    # Swap the sequence length and num attention heads dimensions
    query = query.transpose(0, 2, 1, 3)
    key = key.transpose(0, 2, 1, 3)
    value = value.transpose(0, 2, 1, 3)

    # Apply rotational embeddings
    query, key = self.rotational_embedding.forward(position_ids, query, key)

    # Support GQA
    if self.config["num_attention_heads"] != self.config["num_key_value_heads"]:
      # Repeat the key and value vectors to match the number of attention heads
      n_repeat = self.config["num_attention_heads"] // self.config["num_key_value_heads"]
      key = key.repeat(n_repeat, axis=2)
      value = value.repeat(n_repeat, axis=2)
    
    # Output is (batch_size, num_attention_heads, seq_len, seq_len)
    attention_scores = query @ key.transpose(0, 1, 3, 2) / math.sqrt(self.head_size)

    if attention_mask is not None:
      attention_scores = attention_scores + attention_mask
    
    attention_probs = mx.softmax(attention_scores, axis=-1)

    # Output is (batch_size, num_attention_heads, seq_len, head_size)
    context = attention_probs @ value

    # Go back to (batch_size, seq_len, hidden_size)
    context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.config["hidden_size"])

    return self.output_proj.forward(context)


class LlamaMLP:
  def __init__(self, layer_idx, config):
    self.layer_idx = layer_idx
    self.config = config
    self.gate_proj = LinearLayer(f"model.layers.{self.layer_idx}.mlp.gate_proj.weight")
    self.up_proj = LinearLayer(f"model.layers.{self.layer_idx}.mlp.up_proj.weight")
    self.down_proj = LinearLayer(f"model.layers.{self.layer_idx}.mlp.down_proj.weight")

  def load(self):
    self.gate_proj.load()
    self.up_proj.load()
    self.down_proj.load()

  def forward(self, x):
    gate = self.gate_proj.forward(x)
    gate = gate * mx.sigmoid(gate)

    up = gate * self.up_proj.forward(x)
    return self.down_proj.forward(up)


class RMSNorm:
  def __init__(self, weight_name, config):
    self.config = config
    self.weight_name = weight_name
    self.variance_epsilon = config["rms_norm_eps"]

  def load(self):
    self.weight = load_weight_tensor(self.weight_name)

  def forward(self, x: mx.array):
    variance = mx.mean(x ** 2, axis=-1, keepdims=True)
    x = x * mx.rsqrt(variance + self.variance_epsilon)

    return self.weight * x


class LlamaDecoderLayer:
  def __init__(self, layer_idx, config):
    self.layer_idx = layer_idx
    self.config = config
    self.input_norm = RMSNorm(
      f"model.layers.{layer_idx}.input_layernorm.weight",
      config,
    )
    self.attention = LlamaAttention(layer_idx, config)
    self.post_attention_norm = RMSNorm(
      f"model.layers.{layer_idx}.post_attention_layernorm.weight",
      config,
    )
    self.mlp = LlamaMLP(layer_idx, config)

  def load(self):
    self.input_norm.load()
    self.attention.load()
    self.post_attention_norm.load()
    self.mlp.load()
  
  def forward(self, x, position_ids, attention_mask):
    residual = x
    
    x = self.input_norm.forward(x)
    x = self.attention.forward(x, position_ids, attention_mask)
    
    x = x + residual

    residual = x

    x = self.post_attention_norm.forward(x)
    x = self.mlp.forward(x)

    x = x + residual

    return (x,)


class LlamaEmbedding:
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
    self.layers = [LlamaDecoderLayer(i, config) for i in range(config["num_hidden_layers"])]
    self.post_decoder_norm = RMSNorm("model.norm.weight", config)
    self.score_layer = LinearLayer("lm_head.weight")

    self.causal_mask = mx.full((config["max_position_embeddings"], config["max_position_embeddings"]), vals=True, dtype=mx.bool_)
    self.causal_mask = mx.triu(self.causal_mask, k=1)
  
  def load(self):
    self.embed_tokens.load()
    self.post_decoder_norm.load()
    self.score_layer.load()
    for layer in self.layers:
      layer.load()
  
  def forward(self, input_ids, position_ids, attention_mask):
    x = self.embed_tokens.forward(input_ids)

    MIN_FLOAT = -1e30
    causal_mask = self.causal_mask * MIN_FLOAT
    causal_mask = mx.where(mx.isnan(causal_mask), 0, causal_mask)

    causal_mask = causal_mask[None, None, :, :]
    causal_mask = mx.repeat(causal_mask, repeats=input_ids.shape[0], axis=0)

    padding_mask = mx.equal(causal_mask[..., :attention_mask.shape[-1]], 0) * mx.equal(attention_mask[:, None, None, :], 0)
    causal_mask = mx.where(padding_mask, MIN_FLOAT, causal_mask[..., :attention_mask.shape[-1]])

    for layer in self.layers:
      outputs = layer.forward(x, position_ids, causal_mask)
      x = outputs[0]
      LOG.info(f"Completed layer {layer.layer_idx}.")
    
    x = self.post_decoder_norm.forward(x)
    logits = self.score_layer.forward(x)

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
  
  input_ids = mx.array(input_ids.numpy(), dtype=mx.int32)
  attention_mask = mx.array(attention_mask.numpy(), dtype=mx.int32)

  # Apply the position ids.
  position_ids = mx.arange(max_sequence_length, dtype=mx.int32)
  position_ids = mx.where(attention_mask[0] == 0, 1, position_ids)

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
    next_token = mx.argmax(logits, axis=-1)[0, current_position_id].astype(dtype=mx.int32)

    # Put the next token at the end of the input sequence.
    next_pos = current_position_id + 1
    sequence = input_ids[0]
    sequence[next_pos] = next_token
    input_ids[0] = sequence

    # Update the attention mask.
    sequence = attention_mask[0]
    sequence[next_pos] = 1
    attention_mask[0] = sequence

    # Update the position ids.
    sequence = position_ids[0]
    sequence[next_pos] = position_ids[0][current_position_id] + 1
    position_ids[0] = sequence

    # If the next token is the end of sequence token, stop.
    if next_token == tokenizer.eos_token_id:
      break

    current_position_id = next_pos

    LOG.info(f"Current output: {tokenizer.decode(torch.tensor(np.array(input_ids[0])), skip_special_tokens=True)}")

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


if __name__ == '__main__':
  fire.Fire(CLIWrapper)
