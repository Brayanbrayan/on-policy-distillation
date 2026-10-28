# Setup notes (measured)

Hardware: Colab TPU v5e, 1 chip, 15.75 GiB HBM limit.
Stack: jax 0.11.2, libtpu 0.0.48.1, google-tunix 0.1.7, flax 0.12.10.

- Gemma 3 cannot be loaded from Hugging Face by Tunix 0.1.7 (GCS only). Gemma 3 4B and 1B tokenizers verified identical (vocab 262145).
- Pair in use: Qwen3-1.7B teacher, Qwen3-0.6B student (identical tokenizer, 151669 entries; model embedding rows 151936). Weights load as bf16.
- HBM: student 1.11 GiB, teacher 3.21 GiB, both resident 4.32 GiB.
- Model config reports dtype float32 for compute; sampler dtype float32.
- Rollout config chosen: batch 32, cache_size 512, temperature 1.0: about 2627 tok/s steady (peak 8.40 GiB); bs=16 about 1713 tok/s (peak 6.36 GiB). bs=64 fails to load the decode program (7.26 GB reservation scales with batch x cache_size = 32768 positions).
- Tunix Sampler default temperature is 0.0 (greedy): always pass temperature=1.0 for rollouts.
- Sampler output exposes tokens, logprobs, logits, text. Sampler logprobs vs compute_per_token_logps on the same sequences: mean abs diff about 0.002 per token, max about 0.05.
- Sampler strips the stop token from `tokens` (seen in one short test): append <|im_end|> to sequences that ended early before scoring.
- Workflow: Colab is the only writer to this repo; push with save() using a fine-grained token in Colab Secrets.
