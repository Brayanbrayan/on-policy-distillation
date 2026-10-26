# On-policy distillation on a TPU

Small-scale reproduction of on-policy distillation (student samples, teacher scores every
sampled token with a per-token reverse KL used as the advantage), compared against SFT on
teacher samples and GRPO on the same budget. Runs on a single Colab TPU v5e chip using Tunix (JAX).

## Layout
- `src/`        library code (rollouts, teacher scoring, loss, eval)
- `scripts/`    resumable entry points, one per method (sft, grpo, opd)
- `notebooks/`  thin Colab drivers that clone this repo and call scripts
- `configs/`    per-run configs
- `results/`    small metric files (JSON/CSV) only; checkpoints live on Drive/GCS

## Status
Setup in progress.
