"""SFT baseline: LoRA fine-tune the student on teacher samples, then run the GSM8K exam.

Colab usage, after the Setup cell on a TPU runtime:
    %run scripts/train_sft.py
    %run scripts/train_sft.py --seed 1 --run-dir /content/drive/MyDrive/opd-runs/sft_lora_r16_seed1
This script does not touch git; call save(...) from the notebook afterwards.
"""
import argparse, json, os, time

from src.models import make_mesh, hbm, load, STUDENT_ID   # imports src.env_setup first (torch autoload off)
from src.lora import apply_lora
from src import eval_gsm8k as ev

import numpy as np, optax, jax.numpy as jnp
import orbax.checkpoint as ocp
from transformers import AutoTokenizer
from tunix.sft import peft_trainer, metrics_logger, utils as sft_utils
from tunix.generate import sampler as sampler_lib, tokenizer_adapter


def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default="/content/drive/MyDrive/opd-runs/sft_lora_r16")
    p.add_argument("--data", default="/content/drive/MyDrive/opd-runs/sft_data/teacher_samples.jsonl")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--mb", type=int, default=4, help="micro-batch size; use 2 with --acc 16 if memory is tight")
    p.add_argument("--acc", type=int, default=8, help="gradient accumulation steps")
    p.add_argument("--seq-len", type=int, default=640)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=float, default=32.0)
    p.add_argument("--seed", type=int, default=None, help="LoRA init and data order; None reproduces the first run")
    p.add_argument("--eval-n", type=int, default=192)
    p.add_argument("--results", default="results/sft_gsm8k.json")
    return p.parse_args(argv)


def main(a):
    if a.run_dir.startswith("/content/drive") or a.data.startswith("/content/drive"):
        from google.colab import drive; drive.mount("/content/drive")
    os.makedirs(a.run_dir, exist_ok=True)
    json.dump(vars(a), open(f"{a.run_dir}/args.json", "w"), indent=2)

    rows = [json.loads(l) for l in open(a.data)]
    hf_tok = AutoTokenizer.from_pretrained(STUDENT_ID)
    PAD = hf_tok.convert_tokens_to_ids("<|endoftext|>"); IM_END = hf_tok.convert_tokens_to_ids("<|im_end|>")
    L, MB, ACC, EPOCHS, LR = a.seq_len, a.mb, a.acc, a.epochs, a.lr

    assert all(r["completion_ids"][-1] == IM_END for r in rows if r["ended"]), "stop token missing on finished samples"

    # data: fixed-length rows; loss mask = 1 only on the teacher's answer tokens (including the stop token)
    def build(r):
        p = hf_tok.encode(r["prompt"], add_special_tokens=False); c = r["completion_ids"]
        ids = (p + c)[:L]; mask = ([0] * len(p) + [1] * len(c))[:L]
        pad = L - len(ids)
        return np.array(ids + [PAD] * pad, np.int32), np.array(mask + [0] * pad, np.int32)
    X, M = map(np.stack, zip(*[build(r) for r in rows]))
    print("example, last masked tokens:", repr(hf_tok.decode(X[0][M[0] == 1][-12:])))
    print("example, first masked token:", repr(hf_tok.decode(X[0][M[0] == 1][:6])))
    n_opt_steps = (len(rows) // (MB * ACC)) * EPOCHS

    def batches():
        for ep in range(EPOCHS):
            order = np.random.RandomState(ep if a.seed is None else a.seed * 1000 + ep).permutation(len(rows))  # deterministic, so resume can replay it
            for i in range(0, len(order) - MB + 1, MB):
                idx = order[i:i + MB]
                yield peft_trainer.TrainingInput(input_tokens=X[idx], input_mask=M[idx])
    print(len(rows), "samples ->", n_opt_steps, "optimizer steps | loss tokens per epoch:", int(M.sum()))

    mesh = make_mesh(); base = load(STUDENT_ID, mesh)
    lora = apply_lora(base, mesh, rank=a.rank, alpha=a.alpha, seed=a.seed)
    print("lora enabled:", sft_utils.is_lora_enabled(lora)); hbm("lora model")

    def gen_input(x):
        pm = x.input_tokens != PAD
        return {"input_tokens": x.input_tokens, "input_mask": x.input_mask,
                "positions": sft_utils.build_positions_from_mask(pm), "attention_mask": sft_utils.make_causal_attn_mask(pm)}

    def sft_loss(model, input_tokens, input_mask, positions, attention_mask, images=None):
        logits, _ = model(input_tokens, positions, None, attention_mask)
        ce = optax.softmax_cross_entropy_with_integer_labels(logits[:, :-1, :].astype(jnp.float32), input_tokens[:, 1:])
        m = input_mask[:, 1:].astype(jnp.float32)
        return (ce * m).sum() / (m.sum() + 1e-8)

    sched = optax.warmup_cosine_decay_schedule(0.0, LR, 10, n_opt_steps, LR * 0.1)
    cfg = peft_trainer.TrainingConfig(
        eval_every_n_steps=10**9, max_steps=n_opt_steps, gradient_accumulation_steps=ACC,
        checkpoint_root_directory=f"{a.run_dir}/ckpt",
        checkpointing_options=ocp.CheckpointManagerOptions(save_interval_steps=20, max_to_keep=2),
        metrics_logging_options=metrics_logger.MetricsLoggerOptions(log_dir=f"{a.run_dir}/tb", flush_every_n_steps=5))
    trainer = peft_trainer.PeftTrainer(lora, optax.adamw(sched), cfg).with_gen_model_input_fn(gen_input).with_loss_fn(sft_loss)
    t0 = time.time()
    with mesh:
        trainer.train(batches())
    train_seconds = round(time.time() - t0)       # covers only this run if it resumed from a checkpoint
    hbm("after training")

    # accuracy on the same exam as the zero-shot baselines
    try:
        tok = tokenizer_adapter.TokenizerAdapter(hf_tok); c = lora.config
        smp = sampler_lib.Sampler(transformer=lora, tokenizer=tok, cache_config=sampler_lib.CacheConfig(
            cache_size=640, num_layers=c.num_layers, num_kv_heads=c.num_kv_heads, head_dim=c.head_dim))
        res = ev.evaluate(smp, hf_tok, [IM_END, PAD], ev.load_gsm8k_test(a.eval_n))
        res.update({"train_seconds": train_seconds, "optimizer_steps": n_opt_steps})
        print("SFT student:", res)
        os.makedirs(os.path.dirname(a.results) or ".", exist_ok=True)
        key = f"SFT-LoRA ({sum(r['n_gen'] for r in rows) / 1e6:.1f}M teacher tokens)"
        json.dump({key: res}, open(a.results, "w"), indent=2)
        return res
    except Exception as e:
        print("Training finished and checkpoints are on Drive, but the eval failed:", type(e).__name__, e)


if __name__ == "__main__":
    main(parse())
