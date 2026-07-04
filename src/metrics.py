"""Reusable metric functions for the MIND project."""
import numpy as np

def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return (2 * np.sum(idx * x) / (n * x.sum())) - (n + 1) / n

def gini_ci(values, n_boot=1000):
    vals = np.asarray(values, dtype=float)
    boots = [gini(np.random.choice(vals, size=len(vals), replace=True))
             for _ in range(n_boot)]
    return np.percentile(boots, [2.5, 97.5])

def gini_diff_ci(df, col_a, col_b, n_boot=2000):
    a = df[col_a].values.astype(float)
    b = df[col_b].values.astype(float)
    n = len(df)
    diffs = [gini(a[idx]) - gini(b[idx])
             for idx in (np.random.randint(0, n, n) for _ in range(n_boot))]
    return np.mean(diffs), np.percentile(diffs, [2.5, 97.5])