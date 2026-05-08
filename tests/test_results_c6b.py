import pandas as pd
import numpy as np
from scipy import stats

# Manually enter your table data from the Excel
baseline = {
    0: 228.05, 1: 245.19, 2: 241.28, 3: 219.65, 4: 234.01,
    5: 229.81, 6: 237.37, 7: 243.39, 8: 223.44, 9: 225.02,
    10: 249.45, 11: 232.74, 12: 233.35, 13: 233.66, 14: 224.36,
    15: 247.32, 16: 249.58, 17: 239.13, 18: 210.45, 19: 240.40,
    20: 223.17, 21: 232.16, 22: 236.84, 23: 219.43, 24: 228.70,
    25: 247.10, 26: 238.11, 27: 238.90,
}

c6b = {
    0: 234.46, 1: 238.13, 2: 232.21, 3: 220.44, 4: 248.98,
    5: 238.22, 6: 242.87, 7: 240.22, 8: 218.24, 9: 223.05,
    10: 237.08, 11: 233.24, 12: 233.82, 13: 239.05, 14: 229.47,
    15: 246.42, 16: 251.10, 17: 243.24, 18: 214.26, 19: 239.98,
    20: 234.84, 21: 230.69, 22: 236.66, 23: 198.75, 24: 225.90,
    25: 236.93, 26: 241.74, 27: 240.91,
}

common = sorted(set(baseline.keys()) & set(c6b.keys()))
b_vals = np.array([baseline[f] for f in common])
c_vals = np.array([c6b[f] for f in common])
diffs = c_vals - b_vals

print(f"n={len(common)} folds")
print(f"Baseline mean: {b_vals.mean():.2f} ± {b_vals.std():.2f}")
print(f"C6b mean:      {c_vals.mean():.2f} ± {c_vals.std():.2f}")
print(f"Mean delta:    {diffs.mean():+.2f}")
print(f"Positive folds: {(diffs>0).sum()}/{len(diffs)}")
print(f"Negative folds: {(diffs<0).sum()}/{len(diffs)}")

t, p = stats.ttest_rel(c_vals, b_vals)
print(f"Paired t-test: t={t:.3f}, p={p:.4f}")

print("\nPer-fold breakdown:")
for f, d in zip(common, diffs):
    marker = "✓" if d > 0 else "✗"
    print(f"  Fold {f:2d}: {d:+7.2f}  {marker}")