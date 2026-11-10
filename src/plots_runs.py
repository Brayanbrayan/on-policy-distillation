import json, math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _jl(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []

def run_dir(name):
    for base in ("results/runs", "results/archive_failed", "results/archive_greedy_rollouts"):
        if os.path.isdir(f"{base}/{name}"):
            return f"{base}/{name}"
    raise FileNotFoundError(name)

def _roll(y, w=10):
    y = np.asarray(y, float)
    return np.convolve(y, np.ones(w) / w, mode="valid") if len(y) >= w else y

def accuracy_vs_tokens(runs, out="results/figures/accuracy_vs_tokens.png"):
    z = json.load(open("results/zero_shot_gsm8k.json")); s0, t0 = z["Qwen3-0.6B"]["pass@1"], z["Qwen3-1.7B"]["pass@1"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axhline(t0 * 100, ls="--", c="gray"); ax.text(0.01, t0 * 100 + 0.8, "teacher Qwen3-1.7B", color="gray", fontsize=8)
    ax.axhline(s0 * 100, ls=":", c="gray"); ax.text(0.01, s0 * 100 + 0.8, "student before training", color="gray", fontsize=8)
    for label, name in runs.items():
        ev = _jl(f"{run_dir(name)}/eval.jsonl")
        x = [0.0] + [e["cum_gen_tokens"] / 1e6 for e in ev]; y = [s0] + [e["pass@1"] for e in ev]; n = [192] + [e["n"] for e in ev]
        err = [1.96 * math.sqrt(p * (1 - p) / k) * 100 for p, k in zip(y, n)]
        ax.errorbar(x, [v * 100 for v in y], yerr=err, marker="o", capsize=3, label=label)
    ax.set_xlabel("student-generated tokens (millions)"); ax.set_ylabel("GSM8K pass@1 (%)"); ax.legend(fontsize=8)
    ax.set_title("Accuracy against tokens generated (greedy, 192 problems)")
    os.makedirs(os.path.dirname(out), exist_ok=True); fig.tight_layout(); fig.savefig(out, dpi=200); return fig

def kl_curve(name, out="results/figures/reverse_kl.png"):
    m = _jl(f"{run_dir(name)}/metrics.jsonl"); kl = [r["reverse_kl"] for r in m]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(kl) + 1), kl, alpha=0.3, label="per step")
    ax.plot(range(10, len(kl) + 1), _roll(kl), label="10-step mean")
    ax.axhline(0, c="k", lw=0.8); ax.set_xlabel("step"); ax.set_ylabel("mean per-token reverse KL (nats)"); ax.legend()
    ax.set_title("Student-teacher gap on the student's own samples")
    os.makedirs(os.path.dirname(out), exist_ok=True); fig.tight_layout(); fig.savefig(out, dpi=200); return fig

def length_curve(runs, out="results/figures/length_and_finish.png"):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for label, name in runs.items():
        m = _jl(f"{run_dir(name)}/metrics.jsonl")
        for ax, key in zip(axs, ("mean_len", "frac_ended")):
            v = [r[key] for r in m if key in r]
            if v: ax.plot(range(10, len(v) + 1), _roll(v), label=label)
    axs[0].set_ylabel("mean answer length (tokens)"); axs[1].set_ylabel("share of answers that stop on their own")
    for ax in axs: ax.set_xlabel("step"); ax.legend(fontsize=8)
    os.makedirs(os.path.dirname(out), exist_ok=True); fig.tight_layout(); fig.savefig(out, dpi=200); return fig

def make_all():
    runs = {"On-policy distillation (seed 0)": "opd_s0_v2", "GRPO, no finish requirement (collapsed)": "grpo_s0_never_stops"}
    ok = {}
    for k, n in runs.items():
        try: run_dir(n); ok[k] = n
        except FileNotFoundError: print("missing run, skipped:", n)
    accuracy_vs_tokens(ok); length_curve(ok)
    if "opd_s0_v2" in ok.values(): kl_curve("opd_s0_v2")
    print("figures written to results/figures/")
