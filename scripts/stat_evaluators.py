# stat_evaluators.py
import json
import glob
import numpy as np
from scipy import stats

# Charge les benchmarks DeepSeek et GPT-4o-mini (v2 dataset)
files = sorted(glob.glob("evaluations/benchmark_*.json"), reverse=True)

benchmarks = {}
for f in files:
    try:
        with open(f) as fp:
            data = json.load(fp)
        ev   = data.get("evaluator", "?")
        ds   = data.get("dataset", "golden_dataset_v2.json")
        if "v2" in ds or ds == "golden_dataset_v2.json":
            if ev not in benchmarks:
                benchmarks[ev] = data
                print(f"Loaded {ev} : {f}")
    except Exception:
        pass

print(f"\nEvaluators found: {list(benchmarks.keys())}")

# Compare par paires
evaluator_names = list(benchmarks.keys())
metrics = ["context_recall", "faithfulness", "factual_correctness"]

print("\n" + "="*70)
print("  MULTI-EVALUATOR STATISTICAL TESTS")
print("="*70)

for i in range(len(evaluator_names)):
    for j in range(i+1, len(evaluator_names)):
        ev1_name = evaluator_names[i]
        ev2_name = evaluator_names[j]

        ev1_data = benchmarks[ev1_name]["results"][0]["per_q_log"]
        ev2_data = benchmarks[ev2_name]["results"][0]["per_q_log"]

        print(f"\n  {ev1_name} vs {ev2_name}")
        print(f"  {'─'*60}")
        print(f"  {'Metric':<25} {'Δ':>7} {'p(t)':>7} {'p(W)':>7} {'d':>6} {'Sig':>5}")
        print(f"  {'─'*60}")

        for metric in metrics:
            s1 = [q.get(metric, 0.0) for q in ev1_data if metric in q]
            s2 = [q.get(metric, 0.0) for q in ev2_data if metric in q]

            n  = min(len(s1), len(s2))
            s1, s2 = s1[:n], s2[:n]

            delta = np.mean(s1) - np.mean(s2)

            try:
                _, p_t = stats.ttest_rel(s1, s2)
            except Exception:
                p_t = 1.0

            try:
                _, p_w = stats.wilcoxon(s1, s2)
            except Exception:
                p_w = 1.0

            pooled = np.sqrt((np.std(s1, ddof=1)**2 +
                              np.std(s2, ddof=1)**2) / 2)
            d = (np.mean(s1) - np.mean(s2)) / (pooled + 1e-8)

            # Bootstrap CI on delta
            rng = np.random.default_rng(42)
            deltas = []
            for _ in range(1000):
                idx = rng.integers(0, n, size=n)
                deltas.append(
                    np.mean([s1[k] for k in idx]) -
                    np.mean([s2[k] for k in idx])
                )
            ci_lo = np.percentile(deltas, 2.5)
            ci_hi = np.percentile(deltas, 97.5)

            sig = "✓" if p_t < 0.05 else "✗"
            print(f"  {metric:<25} {delta:>+7.3f} {p_t:>7.4f} "
                  f"{p_w:>7.4f} {d:>6.3f} {sig:>5}")
            print(f"  {'':25} Bootstrap CI: [{ci_lo:+.3f}, {ci_hi:+.3f}]")