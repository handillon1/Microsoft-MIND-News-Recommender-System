import numpy as np
from src.eval import ndcg_at_k, mrr
from src.metrics import intra_list_diversity

def _minmax(x):
    x = np.asarray(x, dtype=float)
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)

def mmr_ranker(row, base_score, emb, row_of, lam=0.3):
    cands = [nid for nid, _ in row.parsed]
    n = len(cands)
    base = _minmax([base_score(row, c) for c in cands])
    idx  = [row_of.get(c) for c in cands]
    have = np.array([i is not None for i in idx])
    V = np.zeros((n, emb.shape[1]))
    for r, i in enumerate(idx):
        if i is not None:
            V[r] = emb[i]
    order, chosen = [], np.zeros(n, bool)
    max_sim = np.zeros(n)
    for step in range(n):
        div = np.ones(n) if step == 0 else (1 - max_sim)
        val = (1 - lam) * base + lam * div
        val[chosen] = -np.inf
        j = int(np.argmax(val))
        order.append(cands[j]); chosen[j] = True
        if have[j]:
            sims = V @ V[j]; sims[~have] = 0
            max_sim = np.maximum(max_sim, sims)
    return order

def make_reranker(base_score, emb, row_of, lam=0.3):
    """Factory: returns a ranker(row) with the given lambda and data bound in."""
    def ranker(row):
        return mmr_ranker(row, base_score, emb, row_of, lam=lam)
    return ranker

def evaluate_ranker(ranker, dev, emb, row_of, k=10):
    ndcgs, mrrs, divs = [], [], []
    for row in dev.itertuples(index=False):
        parsed = row.parsed
        if sum(l for _, l in parsed) == 0:
            continue
        order  = ranker(row)
        lab    = dict(parsed)
        labels = [lab[nid] for nid in order]
        ndcgs.append(ndcg_at_k(labels, k))
        mrrs.append(mrr(labels))
        divs.append(intra_list_diversity(order[:k], emb, row_of))
    return {"nDCG@10": np.mean(ndcgs), "MRR": np.mean(mrrs),
            "diversity@10": np.nanmean(divs)}