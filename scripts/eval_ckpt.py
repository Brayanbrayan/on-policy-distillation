"""Greedy GSM8K evaluation of saved LoRA checkpoints on a large test slice, with per-problem results.
Colab:
  %run scripts/eval_ckpt.py --tag opd_s0_v2 --ckpt-dir /content/drive/MyDrive/opd-runs/opd_s0_v2 --steps 0 50 100 230
Step 0 is the untrained student (a fresh LoRA starts at zero, so it equals the base model).
"""
import argparse, json, os, subprocess, sys, time
try:
    import datasets
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "datasets"], check=True)
from src.models import make_mesh, hbm, load, STUDENT_ID   # imports src.env_setup first
from src.lora import apply_lora
from src import eval_gsm8k as ev
from src import opd
import numpy as np
from flax import nnx
from transformers import AutoTokenizer
from tunix.generate import sampler as sampler_lib, tokenizer_adapter


def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--steps", type=int, nargs="+", required=True)
    p.add_argument("--n", type=int, default=1312)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--max-new", type=int, default=384)
    p.add_argument("--max-prompt", type=int, default=256)
    return p.parse_args(argv)


def paired_diff(base_rows, rows, n_boot=2000, seed=0):
    a = np.array([r["correct"] for r in base_rows], float); b = np.array([r["correct"] for r in rows], float)
    d = b - a; rng = np.random.RandomState(seed)
    boots = [d[rng.randint(0, len(d), len(d))].mean() for _ in range(n_boot)]
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main(a):
    from google.colab import drive; drive.mount("/content/drive")
    assert a.n % a.bs == 0
    hf_tok = AutoTokenizer.from_pretrained(STUDENT_ID); tok = tokenizer_adapter.TokenizerAdapter(hf_tok)
    PAD = hf_tok.convert_tokens_to_ids("<|endoftext|>"); IM_END = hf_tok.convert_tokens_to_ids("<|im_end|>")
    eos = [IM_END, PAD]
    exam = ev.load_gsm8k_test(a.n)
    prompts = [ev.make_prompt(hf_tok, q) for q, _ in exam]; golds = [g for _, g in exam]
    assert max(len(hf_tok.encode(p)) for p in prompts) <= a.max_prompt, "a prompt exceeds max-prompt"

    mesh = make_mesh()
    student = apply_lora(load(STUDENT_ID, mesh), mesh, seed=0)
    graphdef, fresh, rest = nnx.split(student, nnx.LoRAParam, ...)
    c = student.config
    smp = sampler_lib.Sampler(transformer=student, tokenizer=tok, cache_config=sampler_lib.CacheConfig(
        cache_size=a.max_prompt + a.max_new, num_layers=c.num_layers, num_kv_heads=c.num_kv_heads, head_dim=c.head_dim))
    os.makedirs("results/final", exist_ok=True)
    allrows, summary = {}, {}
    for step in a.steps:
        params = fresh if step == 0 else opd.load_lora(fresh, f"{a.ckpt_dir}/lora_step{step}.npz")
        nnx.update(student, params)
        t0 = time.time(); rows = []
        for i in range(0, a.n, a.bs):
            out = smp(input_strings=prompts[i:i + a.bs], max_generation_steps=a.max_new, max_prompt_length=a.max_prompt,
                      temperature=0.0, eos_tokens=eos)
            for j, (text, toks) in enumerate(zip(out.text, out.tokens)):
                b = ev.last_boxed(text)
                rows.append({"i": i + j, "correct": bool(ev.is_correct(ev.first_number(b), golds[i + j])),
                             "boxed": b is not None, "n": len(toks), "trunc": len(toks) >= a.max_new})
        allrows[step] = rows
        p = float(np.mean([r["correct"] for r in rows])); half = 1.96 * np.sqrt(p * (1 - p) / len(rows))
        summary[step] = {"n": len(rows), "pass@1": round(p, 4), "ci95_halfwidth": round(float(half), 4),
                         "boxed_rate": round(float(np.mean([r["boxed"] for r in rows])), 4),
                         "truncated_frac": round(float(np.mean([r["trunc"] for r in rows])), 4),
                         "mean_len": round(float(np.mean([r["n"] for r in rows])), 1), "eval_seconds": round(time.time() - t0)}
        if 0 in allrows and step != 0:
            d, lo, hi = paired_diff(allrows[0], rows)
            summary[step]["diff_vs_step0"] = {"mean": round(d, 4), "ci95": [round(lo, 4), round(hi, 4)]}
        print(f"step {step}:", summary[step]); hbm(f"after step {step}")
        with open(f"results/final/{a.tag}_step{step}.jsonl", "w") as f:
            f.write("\n".join(json.dumps(r) for r in rows) + "\n")
    json.dump({"tag": a.tag, "n": a.n, "device": __import__("jax").devices()[0].device_kind, "steps": summary}, open(f"results/final/{a.tag}_summary.json", "w"), indent=2)


if __name__ == "__main__":
    main(parse())
