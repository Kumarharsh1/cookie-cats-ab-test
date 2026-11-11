from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import (
    confint_proportions_2indep,
    proportion_effectsize,
    proportions_ztest,
)

st.set_page_config(page_title="Cookie Cats A/B Test", page_icon="🍪", layout="wide")

DATA_PATH = Path(__file__).parent / "cookie_cats.csv"
REQUIRED = {"userid", "version", "sum_gamerounds", "retention_1", "retention_7"}


@st.cache_data
def load_data(source):
    df = pd.read_csv(source)
    for col in ["retention_1", "retention_7"]:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.lower().map({"true": 1, "false": 0})
        df[col] = df[col].astype(int)
    return df


st.title("🍪 Cookie Cats A/B Test: Does Moving the First Gate Hurt Retention?")
st.caption(
    "Gate at level 30 (control) vs level 40 (variant). "
    "Two-proportion z-tests, confidence intervals, SRM check and a power calculator."
)

# ---------- Sidebar ----------
st.sidebar.header("Settings")
alpha = st.sidebar.selectbox("Significance level (alpha)", [0.01, 0.05, 0.10], index=1)
remove_outliers = st.sidebar.checkbox("Remove extreme outliers (rounds ≥ cutoff)", value=True)
cutoff = st.sidebar.number_input("Outlier cutoff (rounds)", 1000, 100000, 10000, step=1000)
uploaded = st.sidebar.file_uploader("Or upload your own CSV (same columns)", type="csv")

source = uploaded if uploaded is not None else DATA_PATH
if uploaded is None and not DATA_PATH.exists():
    st.error("cookie_cats.csv not found. Upload a CSV in the sidebar.")
    st.stop()

df = load_data(source)
if not REQUIRED.issubset(df.columns):
    st.error(f"CSV must contain columns: {sorted(REQUIRED)}")
    st.stop()

raw_n = len(df)
if remove_outliers:
    df = df[df["sum_gamerounds"] < cutoff]
st.write(f"**{len(df):,}** players analysed ({raw_n - len(df)} removed as outliers).")

A = df[df["version"] == "gate_30"]
B = df[df["version"] == "gate_40"]
if len(A) == 0 or len(B) == 0:
    st.error("The `version` column must contain both `gate_30` and `gate_40`.")
    st.stop()

# ---------- Group summary ----------
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
    width='stretch',
)

# ---------- SRM ----------
st.subheader("2. Randomisation check (sample ratio mismatch)")
counts = np.array([len(A), len(B)])
_, srm_p = stats.chisquare(counts)
if srm_p < 0.01:
    st.warning(f"SRM p-value = {srm_p:.4f}. The split is uneven, so randomisation may be imperfect. Interpret with caution.")
else:
    st.success(f"SRM p-value = {srm_p:.4f}. The split looks consistent with a 50/50 design.")

# ---------- Tests ----------
st.subheader("3. Significance tests (gate_40 minus gate_30)")
rows = []
for metric, label in [("retention_7", "7-day retention"), ("retention_1", "1-day retention")]:
    xa, xb = A[metric].sum(), B[metric].sum()
    na, nb = len(A), len(B)
    z, p = proportions_ztest([xb, xa], [nb, na])
    lo, hi = confint_proportions_2indep(xb, nb, xa, na, compare="diff", alpha=alpha)
    rows.append({
        "Metric": label,
        "gate_30": xa / na,
        "gate_40": xb / nb,
        "Difference (pp)": (xb / nb - xa / na) * 100,
        f"{int((1 - alpha) * 100)}% CI low (pp)": lo * 100,
        f"{int((1 - alpha) * 100)}% CI high (pp)": hi * 100,
        "z-score": z,
        "p-value": p,
    })
res = pd.DataFrame(rows)
res["p-value (Holm adj.)"] = multipletests(res["p-value"], alpha=alpha, method="holm")[1]
res["Significant?"] = np.where(res["p-value (Holm adj.)"] < alpha, "Yes", "No")
st.dataframe(
    res.style.format({
        "gate_30": "{:.2%}", "gate_40": "{:.2%}", "Difference (pp)": "{:+.2f}",
        res.columns[4]: "{:+.2f}", res.columns[5]: "{:+.2f}",
        "z-score": "{:.2f}", "p-value": "{:.4f}", "p-value (Holm adj.)": "{:.4f}",
    }),
    width='stretch',
    hide_index=True,
)

# ---------- Chart ----------
st.subheader("4. Retention by group")
fig, ax = plt.subplots(figsize=(6, 3.5))
width = 0.35
for i, (metric, label) in enumerate([("retention_1", "1-day"), ("retention_7", "7-day")]):
    for j, (name, g) in enumerate([("gate_30", A), ("gate_40", B)]):
        p = g[metric].mean()
        err = 1.96 * np.sqrt(p * (1 - p) / len(g))
        ax.bar(i + (j - 0.5) * width, p * 100, width, yerr=err * 100, capsize=4,
               color="#4c78a8" if j == 0 else "#f58518", label=name if i == 0 else None)
ax.set_xticks([0, 1])
ax.set_xticklabels(["1-day retention", "7-day retention"])
ax.set_ylabel("Retention (%)")
ax.legend()
ax.spines[["top", "right"]].set_visible(False)
st.pyplot(fig)

# ---------- Engagement ----------
st.subheader("5. Engagement (game rounds)")
u, p_mw = stats.mannwhitneyu(A["sum_gamerounds"], B["sum_gamerounds"], alternative="two-sided")
st.write(
    f"Median rounds: gate_30 = **{A['sum_gamerounds'].median():.0f}**, "
    f"gate_40 = **{B['sum_gamerounds'].median():.0f}**. "
    f"Mann-Whitney U p-value = **{p_mw:.4f}** (rounds are heavily skewed, so a non-parametric test is used)."
)

# ---------- Recommendation ----------
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

# ---------- Power calculator ----------
with st.expander("Power calculator: how many players does a test need?"):
    base = st.number_input("Baseline 7-day retention", 0.01, 0.99, float(round(A["retention_7"].mean(), 3)), step=0.01)
    mde = st.slider("Minimum detectable effect (percentage points)", 0.2, 5.0, 1.0, step=0.1)
    power = st.slider("Power", 0.70, 0.95, 0.80, step=0.05)
    es = proportion_effectsize(base + mde / 100, base)
    n_needed = NormalIndPower().solve_power(effect_size=es, alpha=alpha, power=power, ratio=1)
    st.write(f"You need about **{int(np.ceil(n_needed)):,} players per group** to detect a {mde:.1f} pp change.")

st.divider()
st.caption("Data: Kaggle, Mobile Games A/B Testing (Cookie Cats). Built by Kumar Harsh.")
