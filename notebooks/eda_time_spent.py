# %% [markdown]
# # EDA — Time Spent Regression Model
#
# **Goal**: Understand which features best predict `duration_minutes` (time spent on a to-do task).
# **Target variable**: `duration_minutes` (continuous, right-skewed)
# **Problem type**: Regression on log1p(duration_minutes)
#
# Key constraint: only features available at **task logging time** are valid.
# (`date`, `project_code`, `project_name`, `task_description`)
#
# Run interactively with `# %%` cells in VS Code (Python Interactive) or
# execute end-to-end with `python notebooks/eda_time_spent.py`

# %% [markdown]
# ## 0. Setup

# %%
import sys
import io
# Force UTF-8 output to handle Portuguese characters on Windows
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import warnings
warnings.filterwarnings("ignore")

import os
import re
import string
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from scipy.stats import (
    kruskal, mannwhitneyu, spearmanr, shapiro,
    chi2_contingency, f_oneway
)
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# -- NLTK resources (download once) -----------------------------------------
for resource in ["stopwords", "punkt", "punkt_tab", "rslp"]:
    try:
        nltk.data.find(
            f"tokenizers/{resource}" if resource.startswith("punkt")
            else f"corpora/{resource}"
        )
    except LookupError:
        nltk.download(resource, quiet=True)

# -- Paths -------------------------------------------------------------------
ROOT    = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw" / "time_spent_tasks.parquet"
FIG_DIR  = ROOT / "docs" / "figures" / "time_spent"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# -- Plot style --------------------------------------------------------------
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
# Sequential palette for duration buckets (cool → warm = short → long)
BUCKET_PALETTE = {
    "≤2 min":    "#2166AC",
    "3–5 min":   "#74ADD1",
    "6–10 min":  "#ABD9E9",
    "11–30 min": "#FDAE61",
    "31–60 min": "#F46D43",
    ">60 min":   "#D73027",
}

print("Setup complete.")
print(f"Data path : {DATA_RAW}")
print(f"Figures   : {FIG_DIR}")

# %% [markdown]
# ## 1. Load & Initial Inspection

# %%
df_raw = pd.read_parquet(DATA_RAW)
print(f"Raw shape : {df_raw.shape}")
print(f"\nColumns   : {df_raw.columns.tolist()}")
print(f"\nDtypes:\n{df_raw.dtypes}")

# Validate target
assert df_raw["duration_minutes"].notnull().all(), "NULL values in target!"
assert (df_raw["duration_minutes"] > 0).all(),     "Non-positive durations found!"

print("\n--- Null Counts ---")
print(df_raw.isnull().sum())

print("\n--- Basic Stats ---")
print(df_raw["duration_minutes"].describe(percentiles=[.1, .25, .5, .75, .9, .95, .99]))

print(f"\nDate range: {df_raw['date'].min().date()} to {df_raw['date'].max().date()}")
print(f"Unique projects: {df_raw['project_code'].nunique()}")
print(f"Unique task descriptions: {df_raw['task_description'].nunique()}")

# %% [markdown]
# ## 2. Target Variable Deep Analysis

# %%
dur = df_raw["duration_minutes"]

# --- 2.1 Discreteness check -------------------------------------------------
vc = dur.value_counts().sort_index()
top10_vals = vc.head(10)
top10_pct  = (top10_vals / len(df_raw) * 100).round(1)
print("=== Target Value Frequency (top 10) ===")
for v, cnt in top10_vals.items():
    print(f"  {v:>5} min : {cnt:>4} rows  ({top10_pct[v]:.1f}%)")
print(f"\nDistinct values : {dur.nunique()}")
print(f"Top-3 cover     : {(vc.head(3).sum() / len(df_raw) * 100):.1f}% of rows")

# --- 2.2 Normality tests (raw vs log) ---------------------------------------
stat_raw, p_raw   = shapiro(dur.sample(min(500, len(dur)), random_state=42))
log_dur           = np.log1p(dur)
stat_log, p_log   = shapiro(log_dur.sample(min(500, len(log_dur)), random_state=42))
print(f"\nShapiro-Wilk (raw)   : W={stat_raw:.4f}  p={p_raw:.4e}")
print(f"Shapiro-Wilk (log1p) : W={stat_log:.4f}  p={p_log:.4e}")

# --- 2.3 Distribution plots: raw, log, sqrt ----------------------------------
fig, axes = plt.subplots(2, 3, figsize=(15, 9))

# Row 1: raw
axes[0, 0].hist(dur, bins=60, color="#4C72B0", edgecolor="white", linewidth=0.5)
axes[0, 0].set_title("Raw: duration_minutes")
axes[0, 0].set_xlabel("Minutes")
axes[0, 0].set_ylabel("Frequency")

axes[0, 1].hist(dur[dur <= 120], bins=50, color="#4C72B0", edgecolor="white", linewidth=0.5)
axes[0, 1].set_title("Raw (clipped ≤120 min)")
axes[0, 1].set_xlabel("Minutes")

stats.probplot(dur, dist="norm", plot=axes[0, 2])
axes[0, 2].set_title("QQ-Plot (raw)")

# Row 2: log1p
axes[1, 0].hist(log_dur, bins=40, color="#DD8452", edgecolor="white", linewidth=0.5)
axes[1, 0].set_title("Log1p(duration_minutes)")
axes[1, 0].set_xlabel("log1p(minutes)")
axes[1, 0].set_ylabel("Frequency")

axes[1, 1].hist(np.sqrt(dur), bins=40, color="#55A868", edgecolor="white", linewidth=0.5)
axes[1, 1].set_title("Sqrt(duration_minutes)")
axes[1, 1].set_xlabel("sqrt(minutes)")

stats.probplot(log_dur, dist="norm", plot=axes[1, 2])
axes[1, 2].set_title("QQ-Plot (log1p)")

plt.suptitle("Target Distribution: Raw vs Transformed", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(FIG_DIR / "01_target_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: 01_target_distribution.png")

# --- 2.4 Outlier analysis ---------------------------------------------------
thresholds = [30, 60, 120, 240]
print("\n=== Outlier Thresholds ===")
for t in thresholds:
    n = (dur > t).sum()
    print(f"  > {t:>4} min : {n:>4} rows  ({n / len(df_raw) * 100:.1f}%)")

fig, ax = plt.subplots(figsize=(9, 4))
ax.boxplot(dur, vert=False, patch_artist=True,
           boxprops=dict(facecolor="#4C72B0", alpha=0.6),
           medianprops=dict(color="red", linewidth=2))
ax.set_xlabel("duration_minutes")
ax.set_title("Boxplot of duration_minutes — Outlier View")
ax.set_xscale("log")
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
plt.tight_layout()
plt.savefig(FIG_DIR / "02_target_outliers.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: 02_target_outliers.png")

# %% [markdown]
# ## 3. Duration Buckets (for Stratified Analysis)

# %%
def assign_bucket(m):
    if m <= 2:   return "≤2 min"
    if m <= 5:   return "3–5 min"
    if m <= 10:  return "6–10 min"
    if m <= 30:  return "11–30 min"
    if m <= 60:  return "31–60 min"
    return ">60 min"

BUCKET_ORDER = ["≤2 min", "3–5 min", "6–10 min", "11–30 min", "31–60 min", ">60 min"]

df = df_raw.copy()
df["duration_bucket"] = df["duration_minutes"].apply(assign_bucket)
df["log_duration"]    = np.log1p(df["duration_minutes"])

bucket_dist = df["duration_bucket"].value_counts().reindex(BUCKET_ORDER)
print("=== Duration Bucket Distribution ===")
for b, cnt in bucket_dist.items():
    print(f"  {b:<12}: {cnt:>4}  ({cnt / len(df) * 100:.1f}%)")

fig, ax = plt.subplots(figsize=(9, 4))
colors = [BUCKET_PALETTE[b] for b in BUCKET_ORDER]
ax.bar(BUCKET_ORDER, bucket_dist.values, color=colors, edgecolor="white", linewidth=1.2)
for i, (b, v) in enumerate(zip(BUCKET_ORDER, bucket_dist.values)):
    ax.text(i, v + 5, str(v), ha="center", fontweight="bold", fontsize=9)
ax.set_title("Task Count by Duration Bucket")
ax.set_ylabel("Count")
ax.set_xlabel("Duration Bucket")
plt.tight_layout()
plt.savefig(FIG_DIR / "03_duration_buckets.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: 03_duration_buckets.png")

# %% [markdown]
# ## 4. Feature Engineering

# %%
# --- 4.1 Temporal features (from `date`) ------------------------------------
df["day_of_week"] = df["date"].dt.dayofweek         # 0=Mon … 6=Sun
df["month"]       = df["date"].dt.month
df["week_of_year"]= df["date"].dt.isocalendar().week.astype(int)
df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
df["quarter"]     = df["date"].dt.quarter

# --- 4.2 Project features ---------------------------------------------------
df["project_category"] = df["project_name"].apply(
    lambda x: "Personal" if "personal" in str(x).lower() else "Work"
)

# --- 4.3 Text features (from `task_description`) ----------------------------
PT_STOPWORDS = set(stopwords.words("portuguese"))

def clean_text(text: str) -> list[str]:
    """Lowercase, remove punctuation/digits, tokenize, remove PT stopwords."""
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text, language="portuguese")
    return [t for t in tokens if t not in PT_STOPWORDS and len(t) > 2]

df["desc_word_count"] = df["task_description"].str.split().str.len()
df["desc_char_len"]   = df["task_description"].str.len()
df["tokens"]          = df["task_description"].apply(clean_text)
df["desc_clean"]      = df["tokens"].apply(lambda t: " ".join(t))

# has_number: tasks containing numeric quantifiers ("Fazer 10 flexões", "30 abdominais")
df["has_number"] = df["task_description"].str.contains(r"\d", regex=True).astype(int)

# --- 4.4 Repeated task detection (exact match on normalized description) ---
desc_norm = df["task_description"].str.strip().str.lower()
desc_freq = desc_norm.value_counts()
df["task_freq"]       = desc_norm.map(desc_freq)           # how many times this exact task appears
df["is_repeated_task"] = (df["task_freq"] > 1).astype(int) # binary flag

# --- 4.5 Task type classification (heuristic) --------------------------------
def classify_task_type(desc: str) -> str:
    """Heuristic classification of task type based on Portuguese keywords."""
    desc_l = str(desc).lower()
    if re.search(r"\d+\s*(flex|abdominais|polichinelos|agachamentos|burpees|"
                 r"pompas|levantamentos|pull|push|run|corrida|minutos de)", desc_l):
        return "exercise"
    if re.search(r"(ler|leitura|bíblia|orar|oração|devocional|bib)", desc_l):
        return "devotional"
    if re.search(r"(atualizar|update|timesheet|okr|to\s*do|relat)", desc_l):
        return "admin_update"
    if re.search(r"(enviar|email|mensagem|reunião|meeting|apresent|send|"
                 r"ligar|ligar|contato)", desc_l):
        return "communication"
    if re.search(r"(estudar|estudo|aprender|curso|treinamento|aula|pesquis|"
                 r"revisar|review|análise)", desc_l):
        return "study_review"
    if re.search(r"(desenvolver|implementar|codificar|programar|criar|"
                 r"build|deploy|configurar|install)", desc_l):
        return "development"
    if re.search(r"(praticar|tocar|violão|piano|música|instrumento|"
                 r"meditar|meditação)", desc_l):
        return "practice"
    return "other"

df["task_type"] = df["task_description"].apply(classify_task_type)

print("Engineered features added:")
new_cols = [
    "day_of_week", "month", "week_of_year", "is_weekend", "quarter",
    "project_category",
    "desc_word_count", "desc_char_len", "has_number",
    "task_freq", "is_repeated_task", "task_type",
]
for c in new_cols:
    print(f"  {c:<25}: dtype={df[c].dtype}  nulls={df[c].isnull().sum()}  "
          f"unique={df[c].nunique()}")

print(f"\nFinal working shape: {df.shape}")

# %% [markdown]
# ## 5. Project Analysis vs Duration

# %% [markdown]
# ### 5.1 Per-project duration statistics

# %%
proj_stats = (
    df.groupby("project_code")["duration_minutes"]
    .agg(count="count", mean="mean", median="median", std="std",
         q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75),
         p90=lambda x: x.quantile(0.90), max="max")
    .round(1)
    .sort_values("median", ascending=False)
)
proj_stats["cv"] = (proj_stats["std"] / proj_stats["mean"]).round(3)
print("=== Project Duration Statistics ===")
print(proj_stats.to_string())

# Kruskal-Wallis across projects
groups = [g["duration_minutes"].values for _, g in df.groupby("project_code")
          if len(g) >= 5]
stat_kw, p_kw = kruskal(*groups)
print(f"\nKruskal-Wallis H={stat_kw:.2f}  p={p_kw:.4e}  (groups with n≥5)")

# Eta-squared (effect size for KW)
n_total = sum(len(g) for g in groups)
k = len(groups)
eta2_kw = (stat_kw - k + 1) / (n_total - k)
print(f"Eta-squared (KW) = {eta2_kw:.4f}")

# %%
# Boxplots: projects sorted by median duration
proj_order = proj_stats.sort_values("median").index.tolist()

fig, ax = plt.subplots(figsize=(12, 5))
proj_data = [df.loc[df["project_code"] == p, "duration_minutes"].values
             for p in proj_order]
bp = ax.boxplot(proj_data, patch_artist=True, vert=True, showfliers=True,
                labels=proj_order)
for patch in bp["boxes"]:
    patch.set_facecolor("#4C72B0")
    patch.set_alpha(0.6)
for med in bp["medians"]:
    med.set(color="red", linewidth=2)
ax.set_yscale("log")
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
ax.set_ylabel("duration_minutes (log scale)")
ax.set_title(f"Duration by Project  (Kruskal-Wallis p={p_kw:.2e}, η²={eta2_kw:.3f})")
ax.set_xlabel("Project Code")
plt.tight_layout()
plt.savefig(FIG_DIR / "04_project_vs_duration.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: 04_project_vs_duration.png")

# %% [markdown]
# ### 5.2 Project category (Personal vs Work)

# %%
pers = df.loc[df["project_category"] == "Personal", "duration_minutes"]
work = df.loc[df["project_category"] == "Work",     "duration_minutes"]

stat_mw, p_mw = mannwhitneyu(pers, work, alternative="two-sided")
r_effect = stat_mw / (len(pers) * len(work))  # rank-biserial correlation approximation
print("=== Personal vs Work — duration_minutes ===")
print(f"  Personal: n={len(pers)}  median={pers.median():.0f}  mean={pers.mean():.1f}  std={pers.std():.1f}")
print(f"  Work    : n={len(work)}  median={work.median():.0f}  mean={work.mean():.1f}  std={work.std():.1f}")
print(f"  Mann-Whitney U={stat_mw:.0f}  p={p_mw:.4e}  r≈{r_effect:.3f}")

cat_stats = df.groupby("project_category")["duration_minutes"].describe().round(1)
print(f"\n{cat_stats.to_string()}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

sns.boxplot(data=df, x="project_category", y="duration_minutes",
            order=["Personal", "Work"], ax=axes[0],
            palette={"Personal": "#4C72B0", "Work": "#DD8452"})
axes[0].set_yscale("log")
axes[0].yaxis.set_major_formatter(mticker.ScalarFormatter())
axes[0].set_title(f"Duration by Category  (MW p={p_mw:.2e})")
axes[0].set_ylabel("duration_minutes (log scale)")
axes[0].set_xlabel("")

for cat, color in [("Personal", "#4C72B0"), ("Work", "#DD8452")]:
    vals = df.loc[df["project_category"] == cat, "log_duration"]
    axes[1].hist(vals, bins=25, alpha=0.6, label=cat, color=color, density=True)
axes[1].set_title("log1p(duration) Distribution by Category")
axes[1].set_xlabel("log1p(minutes)")
axes[1].set_ylabel("Density")
axes[1].legend()

plt.tight_layout()
plt.savefig(FIG_DIR / "05_category_vs_duration.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: 05_category_vs_duration.png")

# %% [markdown]
# ### 5.3 Duration bucket composition per project

# %%
proj_bucket = (
    df.groupby(["project_code", "duration_bucket"])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=BUCKET_ORDER, fill_value=0)
)
proj_bucket_pct = proj_bucket.div(proj_bucket.sum(axis=1), axis=0) * 100

print("=== % of tasks per duration bucket per project ===")
print(proj_bucket_pct.round(1).to_string())

fig, ax = plt.subplots(figsize=(12, 5))
proj_bucket_pct.plot(kind="bar", stacked=True, ax=ax,
                     color=[BUCKET_PALETTE[b] for b in BUCKET_ORDER],
                     edgecolor="white", linewidth=0.5)
ax.set_title("Duration Bucket Composition by Project")
ax.set_ylabel("% of tasks")
ax.set_xlabel("Project Code")
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
ax.legend(title="Bucket", loc="lower right", bbox_to_anchor=(1.18, 0.1))
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
plt.tight_layout()
plt.savefig(FIG_DIR / "06_project_bucket_composition.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: 06_project_bucket_composition.png")

# %% [markdown]
# ## 6. Temporal Feature Analysis

# %% [markdown]
# ### 6.1 Day of week vs duration

# %%
day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
dow_stats = (
    df.groupby("day_of_week")["duration_minutes"]
    .agg(count="count", mean="mean", median="median", std="std")
    .round(1)
)
dow_stats.index = [day_names[i] for i in dow_stats.index]

groups_dow = [g["duration_minutes"].values for _, g in df.groupby("day_of_week")]
stat_kw_dow, p_kw_dow = kruskal(*groups_dow)
n_dow = len(df)
k_dow = len(groups_dow)
eta2_dow = (stat_kw_dow - k_dow + 1) / (n_dow - k_dow)

print("=== Day of Week Duration Stats ===")
print(dow_stats.to_string())
print(f"\nKruskal-Wallis H={stat_kw_dow:.2f}  p={p_kw_dow:.4e}  η²={eta2_dow:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Median + IQR bar
medians = dow_stats["median"]
means   = dow_stats["mean"]
days_idx = range(len(day_names))
axes[0].bar(day_names, medians.values, color="#4C72B0", alpha=0.7, label="Median")
axes[0].plot(day_names, means.values, "o--", color="#DD8452", label="Mean", linewidth=2)
axes[0].set_title(f"Duration by Day of Week  (KW p={p_kw_dow:.3f}, η²={eta2_dow:.3f})")
axes[0].set_ylabel("duration_minutes")
axes[0].legend()

# Boxplot
grouped_data = [df.loc[df["day_of_week"] == d, "duration_minutes"].values for d in range(7)]
bp = axes[1].boxplot(grouped_data, patch_artist=True, labels=day_names, showfliers=False)
for patch in bp["boxes"]:
    patch.set_facecolor("#4C72B0")
    patch.set_alpha(0.6)
for med in bp["medians"]:
    med.set(color="red", linewidth=2)
axes[1].set_title("Duration Boxplot by Day (no outliers)")
axes[1].set_ylabel("duration_minutes")

plt.tight_layout()
plt.savefig(FIG_DIR / "07_dow_vs_duration.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: 07_dow_vs_duration.png")

# %% [markdown]
# ### 6.2 Month vs duration

# %%
month_stats = (
    df.groupby("month")["duration_minutes"]
    .agg(count="count", mean="mean", median="median", std="std")
    .round(1)
)
groups_month = [g["duration_minutes"].values for _, g in df.groupby("month")]
stat_kw_month, p_kw_month = kruskal(*groups_month)
eta2_month = (stat_kw_month - len(groups_month) + 1) / (len(df) - len(groups_month))

print("=== Month Duration Stats ===")
print(month_stats.to_string())
print(f"\nKruskal-Wallis H={stat_kw_month:.2f}  p={p_kw_month:.4e}  η²={eta2_month:.4f}")

month_labels = {5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Oct",
                11:"Nov",12:"Dec",1:"Jan",2:"Feb",3:"Mar"}
fig, ax = plt.subplots(figsize=(10, 4))
x = month_stats.index
ax.bar([month_labels.get(m, str(m)) for m in x], month_stats["median"].values,
       color="#4C72B0", alpha=0.7, label="Median")
ax.plot([month_labels.get(m, str(m)) for m in x], month_stats["mean"].values,
        "o--", color="#DD8452", label="Mean", linewidth=2)
ax.set_title(f"Duration by Month  (KW p={p_kw_month:.3f}, η²={eta2_month:.3f})")
ax.set_ylabel("duration_minutes")
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "08_month_vs_duration.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: 08_month_vs_duration.png")

# %% [markdown]
# ### 6.3 Weekend vs Weekday

# %%
wknd = df.loc[df["is_weekend"] == 1, "duration_minutes"]
wkdy = df.loc[df["is_weekend"] == 0, "duration_minutes"]
stat_wk, p_wk = mannwhitneyu(wknd, wkdy, alternative="two-sided")
print("=== Weekend vs Weekday ===")
print(f"  Weekday: n={len(wkdy)}  median={wkdy.median():.0f}  mean={wkdy.mean():.1f}")
print(f"  Weekend: n={len(wknd)}  median={wknd.median():.0f}  mean={wknd.mean():.1f}")
print(f"  Mann-Whitney U={stat_wk:.0f}  p={p_wk:.4e}")

# %% [markdown]
# ## 7. Text & Description Features

# %% [markdown]
# ### 7.1 Description length vs duration

# %%
rho_char, p_rho_char = spearmanr(df["desc_char_len"],   df["duration_minutes"])
rho_word, p_rho_word = spearmanr(df["desc_word_count"], df["duration_minutes"])
rho_log_char, _      = spearmanr(df["desc_char_len"],   df["log_duration"])
rho_log_word, _      = spearmanr(df["desc_word_count"], df["log_duration"])

print("=== Description Length Spearman Correlations ===")
print(f"  desc_char_len   vs duration_minutes : ρ={rho_char:.4f}  p={p_rho_char:.4e}")
print(f"  desc_word_count vs duration_minutes : ρ={rho_word:.4f}  p={p_rho_word:.4e}")
print(f"  desc_char_len   vs log_duration     : ρ={rho_log_char:.4f}")
print(f"  desc_word_count vs log_duration     : ρ={rho_log_word:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, xcol, rho, label in [
    (axes[0], "desc_char_len",   rho_char, "Char Length"),
    (axes[1], "desc_word_count", rho_word, "Word Count"),
]:
    sc = ax.scatter(df[xcol], df["log_duration"], alpha=0.25, s=15,
                    c=df["log_duration"], cmap="viridis")
    # Trend line
    z = np.polyfit(df[xcol], df["log_duration"], 1)
    xline = np.linspace(df[xcol].min(), df[xcol].max(), 100)
    ax.plot(xline, np.polyval(z, xline), "r--", linewidth=1.5)
    ax.set_xlabel(label)
    ax.set_ylabel("log1p(duration_minutes)")
    ax.set_title(f"{label} vs log_duration  (ρ={rho:.3f})")

plt.tight_layout()
plt.savefig(FIG_DIR / "09_desc_length_vs_duration.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: 09_desc_length_vs_duration.png")

# %% [markdown]
# ### 7.2 has_number flag vs duration

# %%
num0 = df.loc[df["has_number"] == 0, "duration_minutes"]
num1 = df.loc[df["has_number"] == 1, "duration_minutes"]
stat_num, p_num = mannwhitneyu(num0, num1, alternative="two-sided")

print("=== has_number vs duration ===")
print(f"  No number : n={len(num0)}  median={num0.median():.0f}  mean={num0.mean():.1f}")
print(f"  Has number: n={len(num1)}  median={num1.median():.0f}  mean={num1.mean():.1f}")
print(f"  Mann-Whitney U={stat_num:.0f}  p={p_num:.4e}")

# Per bucket
print("\n  has_number distribution across buckets:")
print(pd.crosstab(df["has_number"], df["duration_bucket"],
                  normalize="index").reindex(columns=BUCKET_ORDER).round(3))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.boxplot(data=df, x=df["has_number"].astype(str), y="duration_minutes", ax=axes[0],
            order=["0", "1"], palette={"0": "#4C72B0", "1": "#DD8452"})
axes[0].set_yscale("log")
axes[0].yaxis.set_major_formatter(mticker.ScalarFormatter())
axes[0].set_xticklabels(["No number", "Has number"])
axes[0].set_title(f"has_number vs duration  (MW p={p_num:.3f})")
axes[0].set_ylabel("duration_minutes (log scale)")

ct = pd.crosstab(df["has_number"], df["duration_bucket"]).reindex(columns=BUCKET_ORDER)
ct.index = ["No number", "Has number"]
ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
ct_pct.plot(kind="bar", stacked=True, ax=axes[1],
            color=[BUCKET_PALETTE[b] for b in BUCKET_ORDER],
            edgecolor="white")
axes[1].set_title("Bucket Distribution: has_number")
axes[1].set_ylabel("% of tasks")
axes[1].yaxis.set_major_formatter(mticker.PercentFormatter())
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
axes[1].legend(title="Bucket", loc="lower right", bbox_to_anchor=(1.28, 0))
plt.tight_layout()
plt.savefig(FIG_DIR / "10_has_number_vs_duration.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: 10_has_number_vs_duration.png")

# %% [markdown]
# ### 7.3 Repeated tasks — consistency analysis

# %%
rep0 = df.loc[df["is_repeated_task"] == 0, "duration_minutes"]
rep1 = df.loc[df["is_repeated_task"] == 1, "duration_minutes"]
stat_rep, p_rep = mannwhitneyu(rep0, rep1, alternative="two-sided")

print("=== is_repeated_task vs duration ===")
print(f"  One-off  (freq=1): n={len(rep0)}  median={rep0.median():.0f}  mean={rep0.mean():.1f}  std={rep0.std():.1f}")
print(f"  Repeated (freq>1): n={len(rep1)}  median={rep1.median():.0f}  mean={rep1.mean():.1f}  std={rep1.std():.1f}")
print(f"  Mann-Whitney U={stat_rep:.0f}  p={p_rep:.4e}")

# Coefficient of Variation per unique task
task_cv = (
    df.groupby("task_description")["duration_minutes"]
    .agg(count="count", mean="mean", std="std", median="median")
    .query("count >= 3")
    .assign(cv=lambda x: x["std"] / x["mean"])
    .sort_values("cv")
)
print(f"\n=== Task Predictability (CV = std/mean, tasks with n≥3) ===")
print(f"  Most predictable (low CV):")
print(task_cv.head(10)[["count","mean","std","cv"]].to_string())
print(f"\n  Least predictable (high CV):")
print(task_cv.tail(10)[["count","mean","std","cv"]].to_string())

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# CV distribution
axes[0].hist(task_cv["cv"].clip(upper=2), bins=30, color="#4C72B0", edgecolor="white")
axes[0].axvline(task_cv["cv"].median(), color="red", linestyle="--",
                label=f"Median CV={task_cv['cv'].median():.2f}")
axes[0].set_title("Coefficient of Variation per Repeated Task (n≥3)")
axes[0].set_xlabel("CV = std / mean")
axes[0].set_ylabel("Task count")
axes[0].legend()

# Top 15 most frequent tasks — median duration
top_tasks = (
    df["task_description"].value_counts().head(15).index
)
top_task_df = df[df["task_description"].isin(top_tasks)].copy()
top_task_df["short_desc"] = top_task_df["task_description"].str[:35]
order_top = (
    top_task_df.groupby("short_desc")["duration_minutes"].median()
    .sort_values().index
)
sns.boxplot(data=top_task_df, y="short_desc", x="duration_minutes",
            order=order_top, ax=axes[1], color="#4C72B0", linewidth=0.8)
axes[1].set_title("Top-15 Most Frequent Tasks — Duration Spread")
axes[1].set_xlabel("duration_minutes")
axes[1].set_ylabel("")

plt.tight_layout()
plt.savefig(FIG_DIR / "11_repeated_tasks_cv.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: 11_repeated_tasks_cv.png")

# %% [markdown]
# ### 7.4 Task type classification vs duration

# %%
tasktype_stats = (
    df.groupby("task_type")["duration_minutes"]
    .agg(count="count", mean="mean", median="median", std="std")
    .round(1)
    .sort_values("median", ascending=False)
)
tasktype_stats["cv"] = (tasktype_stats["std"] / tasktype_stats["mean"]).round(3)
print("=== Task Type Duration Stats ===")
print(tasktype_stats.to_string())

groups_tt = [g["duration_minutes"].values for _, g in df.groupby("task_type")
             if len(g) >= 3]
stat_kw_tt, p_kw_tt = kruskal(*groups_tt)
eta2_tt = (stat_kw_tt - len(groups_tt) + 1) / (len(df) - len(groups_tt))
print(f"\nKruskal-Wallis H={stat_kw_tt:.2f}  p={p_kw_tt:.4e}  η²={eta2_tt:.4f}")

tt_order = tasktype_stats.index.tolist()
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Boxplot
grouped_tt = [df.loc[df["task_type"] == t, "duration_minutes"].values for t in tt_order]
bp = axes[0].boxplot(grouped_tt, patch_artist=True, labels=tt_order, showfliers=False, vert=False)
for patch in bp["boxes"]:
    patch.set_facecolor("#4C72B0")
    patch.set_alpha(0.6)
for med in bp["medians"]:
    med.set(color="red", linewidth=2)
axes[0].set_title(f"Duration by Task Type  (KW p={p_kw_tt:.2e}, η²={eta2_tt:.3f})")
axes[0].set_xlabel("duration_minutes")

# Stacked bucket bar
tt_bucket = (
    df.groupby(["task_type", "duration_bucket"])
    .size().unstack(fill_value=0)
    .reindex(columns=BUCKET_ORDER, fill_value=0)
    .loc[tt_order]
)
tt_bucket_pct = tt_bucket.div(tt_bucket.sum(axis=1), axis=0) * 100
tt_bucket_pct.plot(kind="barh", stacked=True, ax=axes[1],
                   color=[BUCKET_PALETTE[b] for b in BUCKET_ORDER],
                   edgecolor="white")
axes[1].set_title("Bucket Composition by Task Type")
axes[1].set_xlabel("% of tasks")
axes[1].xaxis.set_major_formatter(mticker.PercentFormatter())
axes[1].legend(title="Bucket", loc="lower right", bbox_to_anchor=(1.30, 0))

plt.tight_layout()
plt.savefig(FIG_DIR / "12_task_type_vs_duration.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: 12_task_type_vs_duration.png")

# %% [markdown]
# ### 7.5 Top N-grams by duration bucket

# %%
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for ax, bucket in zip(axes, BUCKET_ORDER):
    texts = df.loc[df["duration_bucket"] == bucket, "desc_clean"]
    all_words = " ".join(texts).split()
    if not all_words:
        ax.set_visible(False)
        continue
    counts = Counter(all_words).most_common(15)
    if not counts:
        ax.set_visible(False)
        continue
    words, freqs = zip(*counts)
    ax.barh(list(reversed(words)), list(reversed(freqs)), color=BUCKET_PALETTE[bucket])
    ax.set_title(f"Top terms — {bucket} ({len(texts)} tasks)")
    ax.set_xlabel("Frequency")

plt.suptitle("Top Unigrams per Duration Bucket", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(FIG_DIR / "13_ngrams_by_bucket.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: 13_ngrams_by_bucket.png")

# %% [markdown]
# ### 7.6 TF-IDF discriminating features (vs log_duration)

# %%
# Remove rows with empty cleaned text
tfidf_mask = df["desc_clean"].str.strip() != ""
df_tfidf = df[tfidf_mask].copy()

tfidf = TfidfVectorizer(max_features=300, ngram_range=(1, 2), min_df=3)
X_tfidf = tfidf.fit_transform(df_tfidf["desc_clean"])
tfidf_feat_df = pd.DataFrame(X_tfidf.toarray(), columns=tfidf.get_feature_names_out())

# Spearman correlation of each TF-IDF feature vs log_duration
sp_results = []
for col in tfidf_feat_df.columns:
    rho, p = spearmanr(df_tfidf["log_duration"].values, tfidf_feat_df[col].values)
    sp_results.append({"feature": col, "rho": rho, "p": p, "abs_rho": abs(rho)})

sp_df = pd.DataFrame(sp_results).sort_values("abs_rho", ascending=False)
print("=== Top 25 TF-IDF features by |Spearman ρ| with log_duration ===")
print(sp_df.head(25).to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 7))
top25 = sp_df.head(25).sort_values("rho")
colors = ["#DD8452" if r > 0 else "#4C72B0" for r in top25["rho"]]
ax.barh(top25["feature"], top25["rho"], color=colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("TF-IDF Features — Spearman ρ with log1p(duration)")
ax.set_xlabel("ρ  (positive = longer tasks)")
plt.tight_layout()
plt.savefig(FIG_DIR / "14_tfidf_spearman.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: 14_tfidf_spearman.png")

# %% [markdown]
# ## 8. Repeated Task Deep-Dive: Exact vs Fuzzy Duration Variance

# %%
# --- 8.1 Exact match: per-task duration consistency -------------------------
task_exact = (
    df.groupby("task_description")["duration_minutes"]
    .agg(count="count", mean="mean", median="median", std="std",
         min="min", max="max")
    .assign(cv=lambda x: (x["std"] / x["mean"]).fillna(0),
            range=lambda x: x["max"] - x["min"])
    .sort_values("count", ascending=False)
)

# Tasks with zero variance (perfectly predictable)
zero_var = task_exact[task_exact["std"] == 0]
high_var  = task_exact[(task_exact["count"] >= 5) & (task_exact["cv"] > 0.5)]

print(f"=== Exact Task Consistency ===")
print(f"  Unique task descriptions: {len(task_exact)}")
print(f"  Tasks with n>=2         : {(task_exact['count'] >= 2).sum()}")
print(f"  Zero-variance tasks (n>=2): {len(zero_var[zero_var['count'] >= 2])}")
print(f"  High-CV tasks (n>=5, CV>0.5): {len(high_var)}")
print(f"\nZero-variance tasks (all same duration):")
print(zero_var[zero_var["count"] >= 3].sort_values("count", ascending=False)
      .head(10)[["count","mean","range"]].to_string())

print(f"\nHigh-variance tasks (CV>0.5, n>=5):")
print(high_var.sort_values("cv", ascending=False)
      .head(10)[["count","mean","std","cv"]].to_string())

# --- 8.2 Exact vs one-off: CV comparison ------------------------------------
# For each row, get the within-task CV (std/mean of all same-description tasks)
df["task_cv"]   = df["task_description"].map(task_exact["cv"].fillna(0))
df["task_count"] = df["task_description"].map(task_exact["count"])

print("\n=== task_cv Spearman correlation with log_duration ===")
rho_cv, p_cv = spearmanr(df["task_cv"], df["log_duration"])
print(f"  ρ={rho_cv:.4f}  p={p_cv:.4e}")

# --- 8.3 task_freq vs log_duration ------------------------------------------
rho_freq, p_freq = spearmanr(df["task_freq"], df["log_duration"])
print(f"\n=== task_freq Spearman correlation with log_duration ===")
print(f"  ρ={rho_freq:.4f}  p={p_freq:.4e}")

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Scatter: task_freq vs log_duration
axes[0].scatter(df["task_freq"], df["log_duration"], alpha=0.15, s=10, color="#4C72B0")
z = np.polyfit(df["task_freq"], df["log_duration"], 1)
xline = np.linspace(df["task_freq"].min(), df["task_freq"].max(), 100)
axes[0].plot(xline, np.polyval(z, xline), "r--", linewidth=1.5)
axes[0].set_xlabel("task_freq (number of occurrences of this exact task)")
axes[0].set_ylabel("log1p(duration_minutes)")
axes[0].set_title(f"Task Frequency vs log_duration  (ρ={rho_freq:.3f})")

# CV by bucket
cv_bucket = df.groupby("duration_bucket")["task_cv"].median().reindex(BUCKET_ORDER)
colors_bkt = [BUCKET_PALETTE[b] for b in BUCKET_ORDER]
axes[1].bar(BUCKET_ORDER, cv_bucket.values, color=colors_bkt, edgecolor="white")
axes[1].set_title("Median task CV (std/mean) per Duration Bucket")
axes[1].set_ylabel("Median task CV")
axes[1].set_xlabel("Duration Bucket")
axes[1].tick_params(axis="x", rotation=20)

plt.tight_layout()
plt.savefig(FIG_DIR / "15_task_freq_cv_analysis.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: 15_task_freq_cv_analysis.png")

# %% [markdown]
# ## 9. Numeric Feature Spearman Correlation Summary

# %%
numeric_feats = [
    "desc_char_len", "desc_word_count", "task_freq", "task_cv",
    "day_of_week", "month", "week_of_year", "is_weekend", "quarter",
    "has_number", "is_repeated_task",
]

print("=== Spearman ρ with log1p(duration_minutes) ===")
spearman_results = []
for feat in numeric_feats:
    vals = df[[feat, "log_duration"]].dropna()
    rho, p = spearmanr(vals[feat], vals["log_duration"])
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    spearman_results.append({"feature": feat, "rho": rho, "p": p, "sig": sig})
    print(f"  {feat:<25}  ρ={rho:+.4f}  p={p:.4e}  {sig}")

sp_summary = pd.DataFrame(spearman_results).sort_values("rho", key=abs, ascending=False)

# %%
# Effect size bar chart
fig, ax = plt.subplots(figsize=(10, 6))
plot_sp = sp_summary.sort_values("rho")
colors = ["#DD8452" if r > 0 else "#4C72B0" for r in plot_sp["rho"]]
ax.barh(plot_sp["feature"], plot_sp["rho"], color=colors)
ax.axvline(0,    color="black",  linewidth=0.8)
ax.axvline(0.1,  color="orange", linewidth=1,   linestyle="--", label="|ρ|=0.1 (small)")
ax.axvline(-0.1, color="orange", linewidth=1,   linestyle="--")
ax.axvline(0.3,  color="red",    linewidth=1,   linestyle="--", label="|ρ|=0.3 (medium)")
ax.axvline(-0.3, color="red",    linewidth=1,   linestyle="--")
ax.set_title("Spearman ρ with log1p(duration_minutes) per Numeric Feature")
ax.set_xlabel("Spearman ρ")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(FIG_DIR / "16_spearman_summary.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: 16_spearman_summary.png")

# %% [markdown]
# ## 10. Outlier Deep-Dive (>60 min)

# %%
outliers = df[df["duration_minutes"] > 60].copy()
print(f"=== Tasks > 60 minutes ({len(outliers)} rows, {len(outliers)/len(df)*100:.1f}% of data) ===")
print("\nBy project:")
print(outliers.groupby("project_code")["duration_minutes"].agg(["count","median","mean","max"]).sort_values("count", ascending=False))
print("\nBy task_type:")
print(outliers.groupby("task_type")["duration_minutes"].agg(["count","median","mean","max"]).sort_values("count", ascending=False))
print("\nTop tasks by duration (>60 min):")
print(outliers[["task_description","project_code","duration_minutes","task_type"]]
      .sort_values("duration_minutes", ascending=False).head(15).to_string(index=False))

# SBM project deep-dive
sbm = df[df["project_code"] == "SBM"].copy()
print(f"\n=== SBM Project Deep-Dive ===")
print(f"  n={len(sbm)}  median={sbm['duration_minutes'].median():.0f}  "
      f"mean={sbm['duration_minutes'].mean():.1f}  max={sbm['duration_minutes'].max():.0f}")
print("  SBM duration value counts:")
print(sbm["duration_minutes"].value_counts().sort_index())

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Outlier distribution by project
proj_out = outliers.groupby("project_code").size().sort_values(ascending=True)
axes[0].barh(proj_out.index, proj_out.values, color="#D73027")
axes[0].set_title("Tasks >60 min by Project")
axes[0].set_xlabel("Count")

# SBM duration distribution
axes[1].hist(sbm["duration_minutes"], bins=30, color="#4C72B0", edgecolor="white")
axes[1].set_title("SBM Project — Duration Distribution")
axes[1].set_xlabel("duration_minutes")
axes[1].set_ylabel("Frequency")

plt.tight_layout()
plt.savefig(FIG_DIR / "17_outlier_analysis.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: 17_outlier_analysis.png")

# %% [markdown]
# ## 11. Bucket vs Exact Regression: Critical Comparison

# %%
print("=== CRITICAL ANALYSIS: Regression vs Bucketed Classification ===\n")

# --- 11.1 Regression target quality ----------------------------------------
# How much variance is explainable? Check inter-task vs intra-task variance.
# If a task always takes the same time, regression is straightforward.
# If duration is essentially random per occurrence, regression adds little value.

task_var_analysis = (
    df.groupby("task_description")["log_duration"]
    .agg(count="count", var="var", mean="mean")
    .query("count >= 2")
)
between_var = task_var_analysis["mean"].var()
within_var  = task_var_analysis["var"].mean()
icc = between_var / (between_var + within_var) if (between_var + within_var) > 0 else 0
print(f"Log-duration variance decomposition (tasks with n≥2):")
print(f"  Between-task variance (μ of task means)  : {between_var:.4f}")
print(f"  Within-task  variance (mean of task vars) : {within_var:.4f}")
print(f"  ICC (intraclass correlation)              : {icc:.4f}")
print(f"  → {icc*100:.1f}% of variance is systematic (between-task), "
      f"{(1-icc)*100:.1f}% is noise (within-task)\n")

# --- 11.2 Bucket classification difficulty analysis -------------------------
print("Bucket classification challenge:")
for bkt in BUCKET_ORDER:
    n = (df["duration_bucket"] == bkt).sum()
    pct = n / len(df) * 100
    print(f"  {bkt:<12}: {n:>4} ({pct:.1f}%)")
print(f"\n  Majority class baseline accuracy: {df['duration_bucket'].value_counts().iloc[0]/len(df)*100:.1f}%")
print(f"  Regression baseline (predict mean): MAE={np.abs(df['duration_minutes'] - df['duration_minutes'].mean()).mean():.1f} min")
print(f"  Regression baseline (predict median): MAE={np.abs(df['duration_minutes'] - df['duration_minutes'].median()).mean():.1f} min")
print(f"  Log-scale MAE (predict log mean): {np.abs(df['log_duration'] - df['log_duration'].mean()).mean():.4f}")

# --- 11.3 Per-project regression feasibility --------------------------------
print("\nPer-project regression difficulty (higher CV = harder):")
for proj in proj_stats.index:
    cv = proj_stats.loc[proj, "cv"]
    n  = proj_stats.loc[proj, "count"]
    print(f"  {proj:<15} n={n:>4}  CV={cv:.2f}  "
          f"{'[easy]' if cv < 0.5 else '[moderate]' if cv < 1.0 else '[hard]'}")

# %% [markdown]
# ## 12. Consolidated Statistical Summary

# %%
# Gather all categorical effect sizes (Kruskal-Wallis eta-squared)
categorical_summary = [
    {"feature": "project_code",     "type": "categorical", "test": "Kruskal-Wallis", "eta2": eta2_kw,    "p": p_kw},
    {"feature": "project_category", "type": "binary",      "test": "Mann-Whitney",   "eta2": None,        "p": p_mw},
    {"feature": "day_of_week",      "type": "categorical", "test": "Kruskal-Wallis", "eta2": eta2_dow,   "p": p_kw_dow},
    {"feature": "month",            "type": "categorical", "test": "Kruskal-Wallis", "eta2": eta2_month, "p": p_kw_month},
    {"feature": "is_weekend",       "type": "binary",      "test": "Mann-Whitney",   "eta2": None,        "p": p_wk},
    {"feature": "has_number",       "type": "binary",      "test": "Mann-Whitney",   "eta2": None,        "p": p_num},
    {"feature": "is_repeated_task", "type": "binary",      "test": "Mann-Whitney",   "eta2": None,        "p": p_rep},
    {"feature": "task_type",        "type": "categorical", "test": "Kruskal-Wallis", "eta2": eta2_tt,    "p": p_kw_tt},
]

print("=== CATEGORICAL FEATURES — Effect Size Summary ===")
for r in sorted(categorical_summary, key=lambda x: x["p"]):
    sig = "***" if r["p"] < 0.001 else "**" if r["p"] < 0.01 else "*" if r["p"] < 0.05 else "n.s."
    eta_str = f"η²={r['eta2']:.4f}" if r["eta2"] is not None else "     —     "
    print(f"  {r['feature']:<20} {r['test']:<18} {eta_str}  p={r['p']:.4e}  {sig}")

print("\n=== CONTINUOUS FEATURES — Spearman ρ with log_duration ===")
print(sp_summary.to_string(index=False))

# %%
# Combined effect size visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Eta-squared for categoricals
cat_plot = [(r["feature"], r["eta2"]) for r in categorical_summary if r["eta2"] is not None]
cat_names, cat_eta2 = zip(*sorted(cat_plot, key=lambda x: x[1]))
axes[0].barh(cat_names, cat_eta2, color="#4C72B0", alpha=0.8)
axes[0].axvline(0.01, color="orange", linestyle="--", linewidth=1, label="Small (0.01)")
axes[0].axvline(0.06, color="red",    linestyle="--", linewidth=1, label="Medium (0.06)")
axes[0].set_title("Categorical Features — Eta-squared (Kruskal-Wallis)")
axes[0].set_xlabel("η² effect size")
axes[0].legend()

# |rho| for continuous
sp_plot = sp_summary.sort_values("rho", key=abs).copy()
sp_colors = ["#DD8452" if r > 0 else "#4C72B0" for r in sp_plot["rho"]]
axes[1].barh(sp_plot["feature"], sp_plot["rho"].abs(), color=sp_colors, alpha=0.8)
axes[1].axvline(0.1, color="orange", linestyle="--", linewidth=1, label="|ρ|=0.1")
axes[1].axvline(0.3, color="red",    linestyle="--", linewidth=1, label="|ρ|=0.3")
axes[1].set_title("Continuous Features — |Spearman ρ| with log_duration")
axes[1].set_xlabel("|ρ|")
axes[1].legend()

plt.suptitle("Feature Effect Sizes Summary", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(FIG_DIR / "18_effect_sizes_summary.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: 18_effect_sizes_summary.png")

# %% [markdown]
# ## 12b. Additional Feature Engineering (from error analysis, iteration 2)
#
# Based on error analysis of the 50-trial Model 3 regression:
# - Non-repeated tasks have 13× higher MAE than repeated tasks
# - SBM project has extreme variance (std=74min, max=750min)
# - Model collapses to ~5-15min predictions for tasks > 15min
# - task_type × is_repeated interaction captures 6.6× duration ratio (admin_update)
# - Descriptions containing time references ("10 minutos") embed duration hints
# - log1p(task_freq) compresses wide 1-200 range for better gradient splits

# %%
# --- 12b.1 Interaction: task_type × is_repeated_task -------------------------
# Non-repeated admin_update takes 6.6× longer than repeated. This interaction
# captures structural differences that neither feature captures alone.
df["task_type_x_repeated"] = (
    df["task_type"].astype(str) + "_" +
    df["is_repeated_task"].astype(str).map({"1": "rep", "0": "new"})
)
print("task_type_x_repeated value counts:")
print(df["task_type_x_repeated"].value_counts().sort_values(ascending=False).to_string())

# --- 12b.2 is_long_project (SBM, ACELEN, PHD: mean > 25 min) ----------------
# These projects have high duration variance and systematically under-predicted.
LONG_PROJECTS = {"SBM", "ACELEN", "PHD"}
df["is_long_project"] = df["project_code"].isin(LONG_PROJECTS).astype(int)
print(f"\nis_long_project distribution: {df['is_long_project'].value_counts().to_dict()}")

# --- 12b.3 log_task_freq (log1p of task_freq) --------------------------------
# Compresses wide range [1, 200] — frequency=1 tasks have much higher error.
df["log_task_freq"] = np.log1p(df["task_freq"])
print(f"\nlog_task_freq stats:\n{df['log_task_freq'].describe()}")

# --- 12b.4 desc_has_time_ref (time-related words in description) -------------
# Tasks like "Praticar 10 minutos de violão" embed duration hints.
TIME_PATTERN = r"(\d+\s*min|\d+\s*hora|\d+\s*h\b|minuto|hora)"
df["desc_has_time_ref"] = df["task_description"].str.contains(
    TIME_PATTERN, case=False, regex=True
).astype(int)
print(f"\ndesc_has_time_ref distribution: {df['desc_has_time_ref'].value_counts().to_dict()}")
# Correlation with target
rho_time, p_time = spearmanr(df["desc_has_time_ref"], df["log_duration"])
print(f"  desc_has_time_ref vs log_duration: ρ={rho_time:.4f}, p={p_time:.4e}")

# --- 12b.5 task_median_duration (historical median per task_description) -----
# 23 tasks have count >= 2. For single-occurrence tasks, use task_type median.
task_type_median_dur = df.groupby("task_type")["log_duration"].median()
task_desc_median_dur = df.groupby("task_description")["log_duration"].median()
df["task_median_duration"] = df["task_description"].map(task_desc_median_dur)
# Fallback to task_type median for consistency (no missing values in parquet)
print(f"\ntask_median_duration stats:\n{df['task_median_duration'].describe()}")

print(f"\n--- All new features added ({len(df)} rows) ---")
for col in ["task_type_x_repeated", "is_long_project", "log_task_freq",
            "desc_has_time_ref", "task_median_duration"]:
    print(f"  {col:<30}: dtype={df[col].dtype}  nulls={df[col].isnull().sum()}  unique={df[col].nunique()}")

# %% [markdown]
# ## 13. Save Processed Data

# %%
PROCESSED_PATH = ROOT / "data" / "processed" / "eda_time_spent_features.parquet"
PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)

# Keep only modeling-relevant columns; drop raw IDs and redundant text
save_cols = [
    # Target
    "duration_minutes",
    "log_duration",
    # Project features
    "project_code",
    "project_name",
    "project_category",
    # Temporal features
    "date",
    "day_of_week",
    "month",
    "week_of_year",
    "is_weekend",
    "quarter",
    # Text features
    "task_description",
    "desc_clean",
    "desc_char_len",
    "desc_word_count",
    "has_number",
    # Engineered task features
    "task_type",
    "is_repeated_task",
    "task_freq",
    "task_cv",
    # Iteration-2 features (from error analysis)
    "task_type_x_repeated",
    "is_long_project",
    "log_task_freq",
    "desc_has_time_ref",
    "task_median_duration",
    # Analysis
    "duration_bucket",
]

df_save = df[save_cols]
df_save.to_parquet(PROCESSED_PATH, index=False)
print(f"Processed data saved to: {PROCESSED_PATH}")
print(f"Shape: {df_save.shape}")
print(f"Columns: {df_save.columns.tolist()}")

# %% [markdown]
# ## 14. Feature Ranking Summary

# %%
print("\n" + "="*65)
print("FEATURE RANKING SUMMARY — Time Spent Regression Model")
print("="*65)

print("\n--- CATEGORICAL / BINARY (Kruskal-Wallis η² or Mann-Whitney) ---")
for r in sorted(categorical_summary, key=lambda x: x["p"]):
    sig = "***" if r["p"] < 0.001 else "**" if r["p"] < 0.01 else "*" if r["p"] < 0.05 else "n.s."
    eta_str = f"η²={r['eta2']:.4f}" if r["eta2"] is not None else "—"
    print(f"  {r['feature']:<22} {eta_str:<15} p={r['p']:.4e}  {sig}")

print("\n--- CONTINUOUS (Spearman ρ with log_duration) ---")
for _, row in sp_summary.iterrows():
    print(f"  {row['feature']:<25}  ρ={row['rho']:+.4f}  p={row['p']:.4e}  {row['sig']}")

print("\n--- TF-IDF TOP 10 (Spearman ρ with log_duration) ---")
print(sp_df.head(10)[["feature","rho","p"]].to_string(index=False))

print(f"\n--- VARIANCE DECOMPOSITION ---")
print(f"  ICC = {icc:.4f} ({icc*100:.1f}% between-task systematic)")
print(f"  Within-task CV range: {task_cv['cv'].min():.2f} – {task_cv['cv'].max():.2f}")
print(f"  Zero-variance tasks: {len(zero_var[zero_var['count'] >= 2])}")
