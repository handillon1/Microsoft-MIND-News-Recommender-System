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

def intra_list_diversity(news_id_list, emb, row_of):
    import numpy as np
    idx = [row_of[n] for n in news_id_list if n in row_of]
    if len(idx) < 2:
        return np.nan
    V = emb[idx]
    S = V @ V.T
    n = len(idx)
    return 1 - (S.sum() - n) / (n * (n - 1))

def _vecs(news_id_list, emb, row_of):
    idx = [row_of[n] for n in news_id_list if n in row_of]
    return emb[idx] if idx else None


def repetition_rate(news_id_list, emb, row_of, tau=0.9):
    """Share of pairs in the list that are near-duplicates (cosine >= tau)."""
    V = _vecs(news_id_list, emb, row_of)
    if V is None or len(V) < 2:
        return np.nan
    S = V @ V.T
    iu = np.triu_indices(len(V), k=1)
    return float((S[iu] >= tau).mean())


def freshness(news_id_list, shown_at, first_seen, tau_hours=24.0):
    """Mean exp(-age / tau) over the list; age = shown_at - first_seen, in hours."""
    ages = []
    for n in news_id_list:
        fs = first_seen.get(n)
        if fs is None:
            continue
        ages.append(max((shown_at - fs).total_seconds() / 3600.0, 0.0))
    if not ages:
        return np.nan
    return float(np.mean(np.exp(-np.asarray(ages) / tau_hours)))


def category_fairness(exposure_counts):
    """1 - Gini of category exposure shares. 1 = perfectly even exposure."""
    vals = np.asarray(list(exposure_counts.values()), dtype=float)
    if vals.sum() == 0:
        return np.nan
    return float(1 - gini(vals))


def overexposure_ratio(exposure_counts, demand_counts):
    """Most over-exposed category: (exposure share) / (click share). Returns (cat, ratio)."""
    exp_total = sum(exposure_counts.values())
    dem_total = sum(demand_counts.values())
    if not exp_total or not dem_total:
        return None, np.nan
    worst, worst_ratio = None, -np.inf
    for c, e in exposure_counts.items():
        d = demand_counts.get(c, 0) / dem_total
        if d == 0:
            continue
        r = (e / exp_total) / d
        if r > worst_ratio:
            worst, worst_ratio = c, r
    return worst, float(worst_ratio)


def fatigue_per_user(user_category_exposures):
    """user_id -> share of that user's exposures taken by their most-shown category."""
    return {u: max(c.values()) / sum(c.values())
            for u, c in user_category_exposures.items() if sum(c.values())}


def fatigue_index(user_category_exposures):
    """Mean of fatigue_per_user. Higher = more concentrated topic exposure."""
    shares = list(fatigue_per_user(user_category_exposures).values())
    return float(np.mean(shares)) if shares else np.nan