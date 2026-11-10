"""Rebuild metrics.jsonl and eval.jsonl for a run from a saved console log.
Usage: %run scripts/import_log.py --log results/raw_logs/opd_s0_v2.txt --out results/runs/opd_s0_v2"""
import argparse, ast, json, os, re
KEYS = {"rKL": "reverse_kl", "reward": "reward_mean", "signal": "groups_with_signal", "len": "mean_len",
        "ended": "frac_ended", "gnorm": "grad_norm"}

def parse(text):
    metrics, evals = [], []
    for line in text.splitlines():
        m = re.match(r"\s*step\s+(\d+)\s*\|(.*)", line)
        if m:
            rec = {"step": int(m.group(1))}
            for part in m.group(2).split("|"):
                part = part.strip(); mm = re.match(r"(rKL|reward|signal|len|ended|gnorm)\s+(-?[\d.]+)", part)
                if mm: rec[KEYS[mm.group(1)]] = float(mm.group(2))
                for k, pat in (("t_sample", r"sample ([\d.]+)s"), ("t_teacher", r"teacher ([\d.]+)s"), ("t_update", r"update ([\d.]+)s")):
                    m2 = re.search(pat, part)
                    if m2: rec[k] = float(m2.group(1))
            metrics.append(rec)
        elif line.startswith("EVAL "):
            evals.append(ast.literal_eval(line[5:]))
    return metrics, evals

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--log", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); metrics, evals = parse(open(a.log).read())
    os.makedirs(a.out, exist_ok=True)
    open(f"{a.out}/metrics.jsonl", "w").write("\n".join(json.dumps(r) for r in metrics) + "\n")
    if evals: open(f"{a.out}/eval.jsonl", "w").write("\n".join(json.dumps(r) for r in evals) + "\n")
    print(f"{len(metrics)} step lines and {len(evals)} EVAL lines -> {a.out}")
