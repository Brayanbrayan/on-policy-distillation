# Setup notes (measured)

Hardware: Colab TPU v5e, 1 chip, 15.75 GiB HBM limit.
Stack: jax 0.11.2, libtpu 0.0.48.1, google-tunix 0.1.7, flax 0.12.10.

- Gemma 3 cannot be loaded from Hugging Face by Tunix 0.1.7 (GCS only). Gemma 3 4B and 1B tokenizers verified identical (vocab 262145).
- Pair in use: Qwen3-1.7B teacher, Qwen3-0.6B student (identical tokenizer, 151669 entries; model embedding rows 151936). Weights load as bf16.
- HBM: student 1.11 GiB, teacher 3.21 GiB, both resident 4.32 GiB.
- Model config reports dtype float32 for compute; sampler dtype float32.
- Sampling, temperature 1.0, 256 max steps, cache_size 1024: bs=8 about 915 tok/s (peak 8.22 GiB); bs=16 about 1313 tok/s (peak 10.26 GiB); bs=32 failed loading the decode program (7.26 GB needed, 2.68 GB free).
- Tunix Sampler default temperature is 0.0 (greedy): always pass temperature=1.0 for rollouts.
- Sampler output exposes tokens, logprobs, logits, text.
