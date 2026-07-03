import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.db import get_engine
from src.metrics import gini

st.set_page_config(page_title="MIND News Analytics", layout="wide")

@st.cache_resource
def get_conn():
    return get_engine()

@st.cache_data
def run(sql):
    return pd.read_sql(sql, get_conn())

st.title("MIND News Recommendation — Analytics")

tab_overview, tab_conc, tab_cat, tab_time = st.tabs(
    ["Overview", "Concentration", "Categories", "Temporal"]
)

with tab_overview:
    k = run("""SELECT AVG(clicked) AS ctr, COUNT(*) AS impressions,
                      SUM(clicked) AS clicks FROM fact_impressions""").iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Overall CTR", f"{k.ctr:.2%}")
    c2.metric("Impressions", f"{k.impressions:,}")
    c3.metric("Clicks", f"{k.clicks:,}")
    st.caption("Baseline engagement across the full impression log.")

with tab_conc:
    art = run("SELECT news_id, SUM(clicked) AS clicks FROM fact_impressions GROUP BY news_id")
    g = gini(art['clicks'])
    st.metric("Article-level click Gini", f"{g:.3f}")

    # Lorenz curve
    x = np.sort(art['clicks'].astype(float).values)
    y = np.insert(np.cumsum(x) / x.sum(), 0, 0)
    px = np.linspace(0, 1, len(y))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(px, y, label=f"Lorenz (Gini={g:.2f})")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.fill_between(px, y, px, alpha=0.15)
    ax.set_xlabel("Cumulative share of articles")
    ax.set_ylabel("Cumulative share of clicks")
    ax.legend()
    st.pyplot(fig)

with tab_cat:
    cat = run("""SELECT n.category, AVG(f.clicked) AS ctr, SUM(f.clicked) AS clicks
                 FROM fact_impressions f JOIN dim_news n USING (news_id)
                 GROUP BY n.category ORDER BY clicks DESC""")
    metric = st.radio("Show:", ["clicks", "ctr"], horizontal=True)   # a widget!
    st.bar_chart(cat.set_index("category")[metric])

with tab_time:
    hr = run("""SELECT EXTRACT(hour FROM shown_at)::int AS hour,
                       COUNT(*) AS impressions, AVG(clicked) AS ctr
                FROM fact_impressions GROUP BY 1 ORDER BY 1""")
    st.bar_chart(hr.set_index("hour")["impressions"])
    st.line_chart(hr.set_index("hour")["ctr"])