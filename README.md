# ZEnfrence

Toy project to learn about how Transformer LLMs work for inference.

## Usage

Install the requirements:

```bash
poetry install
```

Download the model weights (`Llama-2-7b-chat-hf` is the only one supported at the moment) using the Hugging Face CLI:

```bash
huggingface-cli download \
  meta-llama/Llama-2-7b-chat-hf \
  --include "*.safetensors" \
  --local-dir . \
  --local-dir-use-symlinks False
```

Run inference:

```python
poetry run python3 main.py inference --prompt "What is the meaning of life?"
```

This will be super slow, it is in no way optimized. Takes around 30 seconds/token on my machine (M1 MacBook Pro).


## Roadmap

- [x] Get a model to run inference (Llama-2-7b-chat-hf)
- [ ] Implement top-k sampling
- [ ] Implement top-p (nucleus) sampling
- [ ] Optimize inference (CUDA Kernels, MPS Kernels, batching, sharding, etc.)
- [ ] Implement Flash Attention 2
- [ ] Add support for more models (MoE models, etc.)
- [ ] Implement inference for embedding models
