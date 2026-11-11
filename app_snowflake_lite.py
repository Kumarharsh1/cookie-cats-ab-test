import os
from math import asin, sqrt
from statistics import NormalDist

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Cookie Cats A/B Test", page_icon="🍪", layout="wide")

conn = st.connection("snowflake", ttl=os.getenv("SNOWFLAKE_CONNECTION_TTL"))
session = conn.session()
N = NormalDist()


@st.cache_data
def load_data():
    df = session.table("ML_PROJECT.PUBLIC.COOKIE_CATS").to_pandas()
    df.columns = [c.lower() for c in df.columns]
    for col in ["retention_1", "retention_7"]:
        df[col] = df[col].astype(int)
    return df


# ---------- Statistics helpers (no scipy / statsmodels needed) ----------
def two_prop_test(xa, na, xb, nb, alpha):
    """z-test (pooled) and unpooled 95% CI for difference pb - pa."""
    pa, pb = xa / na, xb / nb
    pool = (xa + xb) / (na + nb)
    z = (pb - pa) / sqrt(pool * (1 - pool) * (1 / na + 1 / nb))
    p = 2 * (1 - N.cdf(abs(z)))
    se = sqrt(pa * (1 - pa) / na + pb * (1 - pb) / nb)
    zc = N.inv_cdf(1 - alpha / 2)
    return pb - pa, (pb - pa) - zc * se, (pb - pa) + zc * se, z, p


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(1.0, running)
    return adj


def mann_whitney_p(a, b):
    allv = pd.concat([a, b], ignore_index=True)
    ranks = allv.rank(method="average")
    na, nb = len(a), len(b)
    n = na + nb
    u = ranks.iloc[:na].sum() - na * (na + 1) / 2
    ties = allv.value_counts().to_numpy().astype(float)
    var = na * nb / 12 * ((n + 1) - (ties**3 - ties).sum() / (n * (n - 1)))
    z = (abs(u - na * nb / 2) - 0.5) / sqrt(var)
    return 2 * (1 - N.cdf(z))


def sample_size_per_group(base, mde_pp, alpha, power):
    h = 2 * asin(sqrt(base + mde_pp / 100)) - 2 * asin(sqrt(base))
    return 2 * ((N.inv_cdf(1 - alpha / 2) + N.inv_cdf(power)) / h) ** 2


# ---------- Page ----------
st.title("🍪 Cookie Cats A/B Test: Does Moving the First Gate Hurt Retention?")
st.caption(
    "Gate at level 30 (control) vs level 40 (variant). "
    "Two-proportion z-tests, confidence intervals, SRM check and a power calculator."
)

st.sidebar.header("Settings")
alpha = st.sidebar.selectbox("Significance level (alpha)", [0.01, 0.05, 0.10], index=1)
remove_outliers = st.sidebar.checkbox("Remove extreme outliers (rounds ≥ cutoff)", value=True)
cutoff = st.sidebar.number_input("Outlier cutoff (rounds)", 1000, 100000, 10000, step=1000)

df = load_data()
raw_n = len(df)
if remove_outliers:
    df = df[df["sum_gamerounds"] < cutoff]
st.write(f"**{len(df):,}** players analysed ({raw_n - len(df)} removed as outliers).")

A = df[df["version"] == "gate_30"]
B = df[df["version"] == "gate_40"]
if len(A) == 0 or len(B) == 0:
    st.error("The `version` column must contain both `gate_30` and `gate_40`.")
    st.stop()

# 1. Summary
st.subheader("1. Group summary")
summary = df.groupby("version").agg(
    players=("userid", "count"),
    retention_1day=("retention_1", "mean"),
    retention_7day=("retention_7", "mean"),
    avg_rounds=("sum_gamerounds", "mean"),
    median_rounds=("sum_gamerounds", "median"),
)
st.dataframe(
    summary.style.format(
        {"players": "{:,.0f}", "retention_1day": "{:.2%}", "retention_7day": "{:.2%}",
         "avg_rounds": "{:.2f}", "median_rounds": "{:.0f}"}
    ),
)

# 2. SRM
st.subheader("2. Randomisation check (sample ratio mismatch)")
na, nb = len(A), len(B)
expected = (na + nb) / 2
chi2 = ((na - expected) ** 2 + (nb - expected) ** 2) / expected
srm_p = 2 * (1 - N.cdf(sqrt(chi2)))
if srm_p < 0.01:
    st.warning(f"SRM p-value = {srm_p:.4f}. The split is uneven, so randomisation may be imperfect. Interpret with caution.")
else:
    st.success(f"SRM p-value = {srm_p:.4f}. The split looks consistent with a 50/50 design.")

# 3. Tests
st.subheader("3. Significance tests (gate_40 minus gate_30)")
ci = int((1 - alpha) * 100)
rows = []
for metric, label in [("retention_7", "7-day retention"), ("retention_1", "1-day retention")]:
    xa, xb = int(A[metric].sum()), int(B[metric].sum())
    diff, lo, hi, z, p = two_prop_test(xa, na, xb, nb, alpha)
    rows.append({
        "Metric": label,
        "gate_30": xa / na,
        "gate_40": xb / nb,
        "Difference (pp)": diff * 100,
        f"{ci}% CI low (pp)": lo * 100,
        f"{ci}% CI high (pp)": hi * 100,
        "z-score": z,
        "p-value": p,
    })
res = pd.DataFrame(rows)
res["p-value (Holm adj.)"] = holm(res["p-value"].to_numpy())
res["Significant?"] = np.where(res["p-value (Holm adj.)"] < alpha, "Yes", "No")
st.dataframe(
    res.style.format({
        "gate_30": "{:.2%}", "gate_40": "{:.2%}", "Difference (pp)": "{:+.2f}",
        res.columns[4]: "{:+.2f}", res.columns[5]: "{:+.2f}",
        "z-score": "{:.2f}", "p-value": "{:.4f}", "p-value (Holm adj.)": "{:.4f}",
    }),
    hide_index=True,
)

# 4. Charts
st.subheader("4. Retention by group (%)")
c1, c2 = st.columns(2)
with c1:
    st.write("1-day retention")
    st.bar_chart(pd.DataFrame({"retention %": [A["retention_1"].mean() * 100, B["retention_1"].mean() * 100]},
                              index=["gate_30", "gate_40"]))
with c2:
    st.write("7-day retention")
    st.bar_chart(pd.DataFrame({"retention %": [A["retention_7"].mean() * 100, B["retention_7"].mean() * 100]},
                              index=["gate_30", "gate_40"]))

# 5. Engagement
st.subheader("5. Engagement (game rounds)")
p_mw = mann_whitney_p(A["sum_gamerounds"], B["sum_gamerounds"])
st.write(
    f"Median rounds: gate_30 = **{A['sum_gamerounds'].median():.0f}**, "
    f"gate_40 = **{B['sum_gamerounds'].median():.0f}**. "
    f"Mann-Whitney U p-value = **{p_mw:.4f}** (rounds are heavily skewed, so a rank-based test is used)."
)

# 6. Recommendation
st.subheader("6. Recommendation")
r7 = res.iloc[0]
if r7["Significant?"] == "Yes" and r7["Difference (pp)"] < 0:
    st.info(
        f"**Keep the gate at level 30.** Moving it to level 40 lowered 7-day retention by "
        f"{abs(r7['Difference (pp)']):.2f} percentage points, a statistically significant drop, "
        f"with no engagement benefit to offset it."
    )
elif r7["Significant?"] == "Yes":
    st.info("**Consider moving the gate to level 40.** 7-day retention improved significantly.")
else:
    st.info("**No significant difference in 7-day retention.** The data does not justify changing the gate.")
st.caption("Caveats: only retention and rounds were measured (no revenue); single test period; check the SRM result above.")

# Power calculator
with st.expander("Power calculator: how many players does a test need?"):
    base = st.number_input("Baseline 7-day retention", 0.01, 0.90, float(round(A["retention_7"].mean(), 3)), step=0.01)
    mde = st.slider("Minimum detectable effect (percentage points)", 0.2, 5.0, 1.0, step=0.1)
    power = st.slider("Power", 0.70, 0.95, 0.80, step=0.05)
    n_needed = sample_size_per_group(base, mde, alpha, power)
    st.write(f"You need about **{int(np.ceil(n_needed)):,} players per group** to detect a {mde:.1f} pp change.")

st.divider()
st.caption("Running natively in Snowflake. Data: Kaggle, Mobile Games A/B Testing (Cookie Cats). Built by Kumar Harsh.")
