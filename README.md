# Constraint-Aware News Recommendation System

A personalized news recommender built on the Microsoft MIND dataset that
maximizes engagement (CTR) while balancing diversity, freshness, and
recommendation-fatigue constraints.

## Tech stack
Python, pandas, scikit-learn, Sentence-Transformers, PostgreSQL, MLflow,
Streamlit / Tableau.

## Project structure
- `data/` — raw + processed MIND data (gitignored)
- `notebooks/` — EDA and prototyping
- `src/` — recommenders, metrics, NLP modules
- `sql/` — schema and queries
- `dashboards/` — Streamlit app / Tableau workbooks
- `reports/` — product insights write-up and figures

## Problem Statement

    We want news readers to discover relevant, varied stories without being shown the same articles or topic
    repeatedly. We aim to cultivate an audience with repeat engagement across the time frame. A recommendation system focusing solely on click through rate can ruin recommendations and create feeds focused only on suggesting the most popular articles, encouraging repetition and narrowing topics; A long term retention risk.To balance the trade off, we should quantify the trade off between the demand for an increase in immediate engagement and a diverse recommendation system by identifying the exchange rate between engagement and diversity and recommend where it stops being worth paying. A success will be consistent click through rate, lowered topic repetition and increased intra list diversity, verified with experiment and guardrails. Failures can be identified when recommendations focus solely on few popular articles, users get suggested only a narrow set of topics, and readers are exposed to similar content leads to higher fatigue/churn.

## Exploratory Data Analysis — Key Findings

Full analysis in [`notebooks/03_eda.ipynb`](notebooks/03_eda.ipynb).

### Engagement is extremely concentrated across articles
Full article analysis in [`notebooks/03_article_analysis.ipynb`](notebooks/03_article_analysis.ipynb).

- **Article-level click Gini = 0.94** (95% CI [0.94, 0.95]) — clicks are
  extraordinarily concentrated; a small set of articles captures nearly all
  engagement, with a long tail receiving almost none.
- **Article-level impression Gini = 0.92** — exposure is itself highly
  concentrated, so most click concentration is *exposure-driven*; engagement
  concentrates only modestly further (0.94 vs 0.92).
- **Top 1% of articles capture ~51.6% of all clicks.**

![Article click concentration (Lorenz curve)](reports/figures/article analysis/article_lorenz_curve.png)

### Topic concentration: demand outpaces supply
Full topic analysis in [`notebooks/02_topic_analysis.ipynb`](notebooks/02_topic_analysis.ipynb).

- **Subcategory:** click Gini **0.81** vs. impression Gini **0.76** — a small
  but statistically robust gap (paired bootstrap mean 0.05, 95% CI [0.03, 0.07]):
  engagement concentrates into a narrower set of subcategories than exposure does.
- **Category:** click Gini **0.51** vs. impression Gini **0.48** — moderate
  concentration (coarse, only ~18 categories).

### Product implication
Engagement is concentrated enough that a pure-CTR recommender would reinforce a
narrow popularity loop. This is the empirical case for the constraint-aware
approach: deliberately counterbalancing popularity with diversity and freshness
to surface the long tail without sacrificing engagement.

### User activity & cold-start
Full trend analysis in [`notebooks/04_trend_analysis.ipynb`](notebooks/04_trend_analysis.ipynb).

- 50k users; history is right-skewed (median 11 prior clicks, mean 18.5,
  max 558) — a small set of power users inflates the average.
- **Cold-start:** 1.8% of users have zero history; 26% have ≤5 prior clicks —
  motivating content-based methods alongside collaborative filtering.
- **Repeat exposure is weak at both grains.** Average top-category share is
  ~0.30 vs. the 0.27 global news share; average top-subcategory share is 0.108
  (≥20-impression users) vs. a null-model expectation of 0.096. Per-user topic
  concentration barely exceeds what random draws from the global mix produce —
  so the logged data shows little personalized topic lock-in.
- **Implication:** current exposure is healthily varied per user; the fatigue
  guardrail is therefore *preventive* — ensuring a CTR-optimizing ranker does
  not introduce narrowing — rather than fixing an existing problem. The active
  concentration risks are article-level popularity (Gini 0.94) and supply-side
  category imbalance (news 27%).

### Temporal trends
- **Impressions follow a clear daily cycle** — overnight trough, broad
  mid-morning peak (08:00–12:00) — confirming clean timestamp parsing.
- **CTR varies only modestly (~3.8%–4.3%).** Adding 95% confidence bands
  sharpened the read: the **08:00 peak** and **13:00 midday dip** are precise
  and robust (high volume, narrow bands), while the apparent late-evening
  CTR spike has wide intervals (low volume) and is **not** statistically
  distinguishable from the morning level — a reminder not to over-read
  low-sample hours.
- Window spans ~6 days (Nov 9–14, 2019), ~2x higher weekday volume; too short
  for weekly-seasonality claims.
![CTR by hour of day with 95% CI](reports/figures/trend analysis/temporal_by_hour_ci.png)

## Dashboard

Interactive Streamlit analytics dashboard (overview KPIs, concentration,
categories, temporal trends):

![MIND analytics dashboard](reports/figures/dashboard.png)

## Baseline Recommenders — Benchmark

Baselines evaluated on the MIND-small dev set by ranking each impression's
candidate list (nDCG@10 and MRR averaged over impressions; a random scorer is
included as a floor). Evaluation harness in [`src/eval.py`](src/eval.py),
scoring functions in [`src/recommenders.py`](src/recommenders.py).

| Model | nDCG@10 | MRR |
|-------|:-------:|:---:|
| Random (floor) | 0.287 | 0.247 |
| Popularity | 0.310 | 0.267 |
| **Content-based** | **0.359** | **0.321** |
| Collaborative filtering (item–item) | 0.303 | 0.262 |

**Takeaways**
- **Content based ranking wins** Matching candidates to the
  categories/subcategories in a user's click history beats popularity by
  +0.050 nDCG (paired t-test, p ≈ 0) — roughly double popularity's edge over
  random. Topic preference alone is a strong personalization signal.
- **Collaborative filtering underperforms popularity**, consistent with news
  item cold-start and sparsity: dev candidates are often fresh articles with
  little or no training co-click signal, so item–item similarity is unreliable.
- **Implication:** content signal drives news recommendation. This motivates
  Phase 4 — replacing crude category matching with semantic title/abstract
  embeddings to capture similarity beyond exact category labels.

## Semantic Layer

Article embeddings generated with Sentence-Transformers (`all-MiniLM-L6-v2`,
384-d) over each article's title + abstract (~65k articles). Artifacts in
`models/`; metric in [`src/metrics.py`](src/metrics.py).

- **Validation:** nearest-neighbour retrieval is coherent — a St. Louis
  shooting story returns other St. Louis shootings first, then related crime
  stories elsewhere, capturing both event type and location from text alone.
- **Topic structure beyond categories:** KMeans (k=50) recovers themes finer
  than MIND's 18 categories — e.g. it separates *individual* sports (golf,
  tennis, MMA) from *team* sports/roster news, and splits the broad `news`
  category into Ukraine politics, crime, and disasters.
- **Redundancy:** ~XX% of articles have a near-duplicate at cosine > 0.90,
  including exact duplicates published under different `news_id`s. Threshold
  methods over-flag formulaic headline templates (e.g. race recaps differing
  only by venue), so duplicates are treated as a signal, not ground truth.
- **Diversity metric:** intra-list diversity = 1 − mean pairwise cosine
  similarity. Validated by construction:
  redundant list (one cluster) **0.756** < popularity list **0.861** 
  varied list (ten clusters) **0.955**. Values compress into a high band, as
  expected for text embeddings — differences are interpreted relatively.

## Optimization Objective

The system does not maximize a single metric. It maximizes engagement **subject
to constraints** that protect the long-term experience:

**Maximize:** predicted engagement (click-through rate)

**Subject to:**
- **Diversity** — intra-list diversity stays above a threshold
- **Freshness** — recommended articles stay recent (first-appearance proxy)
- **Fatigue** — per-user repeated topic exposure stays below a limit
- **Category fairness** — no systematic over/under-exposure of topics

**Operationalized** as a re-ranking score applied to each candidate list:

FinalScore = w₁·P(engage) + w₂·freshness + w₃·diversity_gain
             − w₄·fatigue_penalty − w₅·repetition_penalty

**The decision:** the diversity/engagement balance is exposed as a single
tunable weight (λ). The project measures the resulting tradeoff curve and
selects an operating point that maximizes diversity within a bounded,
pre-specified CTR-loss budget — i.e. it estimates the exchange rate between
engagement and diversity and recommends where it stops being worth paying.

## Experimentation

Simulated A/B test (counterfactual offline evaluation on the MIND dev set;
labels fixed, so this measures ranking quality by arm, not live behavior).

- **Design:** users randomized to control (λ=0, pure engagement) vs. treatment
  (λ=0.5, diversity-constrained re-ranker); ~8k users. Randomized by user.
- **Hypothesis:** treatment raises intra-list diversity while keeping
  engagement within a ~1% guardrail.
- **Results (per-user means, Welch t-test):**
  - Diversity: **+0.018 (0.918 → 0.935), p ≈ 1e-84** — significant.
  - Engagement (nDCG@10): **−0.0008, p = 0.90** — no significant change; guardrail held.
- **Decision: ship λ=0.5** — meaningfully more diverse feeds at no significant
  engagement cost.
- **Caveats:** offline/counterfactual (no observed live clicks); independent-
  groups design has wide engagement CIs — a paired design would estimate the
  small engagement cost more precisely.