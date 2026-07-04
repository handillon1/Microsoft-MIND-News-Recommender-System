import numpy as np

def random_recommender(seed=0):
    rng = np.random.default_rng(seed)
    return lambda row, news_id: rng.random()

def popularity_recommender(train_clicks):
    return lambda row, news_id: train_clicks.get(news_id, 0)

def content_recommender(cat_of, subcat_of, train_clicks):
    def score(row, news_id):
        prof = row.profile
        if prof is None:
            return train_clicks.get(news_id, 0)
        c, s = cat_of.get(news_id), subcat_of.get(news_id)
        return (prof["cats"].get(c, 0) + 2 * prof["subs"].get(s, 0)) / prof["total"]
    return score

def cf_recommender(item_vecs, iidx, train_clicks):
    def score(row, news_id):
        j = iidx.get(news_id)
        if j is None or len(row.hist_idx) == 0:
            return train_clicks.get(news_id, 0)
        return float(item_vecs[row.hist_idx].dot(item_vecs[j].T).sum())
    return score