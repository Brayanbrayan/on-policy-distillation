import json, math, os
import matplotlib.pyplot as plt

def zero_shot_bar(results_path="results/zero_shot_gsm8k.json", out_path="results/figures/zero_shot_gsm8k.png"):
    res = json.load(open(results_path))
    names = list(res)
    p = [res[k]["pass@1"] for k in names]
    n = [res[k]["n"] for k in names]
    err = [1.96 * math.sqrt(pi * (1 - pi) / ni) for pi, ni in zip(p, n)]
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(names, [x * 100 for x in p], yerr=[e * 100 for e in err], capsize=6, color=["#4C78A8", "#F58518"])
    for b, pi, e in zip(bars, p, err):
        ax.text(b.get_x() + b.get_width() / 2, (pi + e) * 100 + 1.5, f"{pi * 100:.1f}%", ha="center")
    ax.set_ylabel("GSM8K pass@1 (%)")
    ax.set_ylim(0, 100)
    ax.set_title(f"Zero-shot GSM8K, greedy, n={n[0]}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    return fig
