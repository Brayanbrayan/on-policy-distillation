import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def paired_figure(path="results/final/summary_cap640.json", out="results/figures/final_paired_change.png"):
    d = json.load(open(path))["methods"]
    names = [m["name"] for m in d]; mean = [m["diff"]["mean"] * 100 for m in d]
    lo = [m["diff"]["ci95"][0] * 100 for m in d]; hi = [m["diff"]["ci95"][1] * 100 for m in d]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.3))
    for ax, zoom in zip(axs, (False, True)):
        idx = [i for i, n in enumerate(names) if not (zoom and abs(mean[i]) > 7)]
        y = list(range(len(idx)))
        ax.errorbar([mean[i] for i in idx], y, xerr=[[mean[i] - lo[i] for i in idx], [hi[i] - mean[i] for i in idx]], fmt="o", capsize=4)
        ax.set_yticks(y); ax.set_yticklabels([names[i] for i in idx]); ax.invert_yaxis()
        ax.axvline(0, c="k", lw=0.8); ax.set_xlabel("change in GSM8K accuracy vs untrained student (points)")
        ax.set_title("all methods and the teacher" if not zoom else "zoom: student methods only")
        if zoom: ax.set_xlim(-4, 7)
    fig.suptitle("1,312 test problems, 640-token cap, paired 95% intervals")
    os.makedirs(os.path.dirname(out), exist_ok=True); fig.tight_layout(); fig.savefig(out, dpi=200); return fig
