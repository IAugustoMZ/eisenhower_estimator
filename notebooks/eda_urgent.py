# %% [markdown]
# # EDA — Model 2: Urgent / Not Urgent
#
# **Goal**: Understand which features best predict whether a task is `urgent=1`.
# **Target variable**: `urgent` (binary: 0 = not urgent, 1 = urgent)
#
# **Key constraint**: Only features available at **task creation time** are valid.
# Features like `is_overdue` (requires knowing today's date relative to due_date
# after task creation) are excluded.
#
# Run interactively with `# %%` cells in VS Code (Python Interactive) or
# execute end-to-end with `python notebooks/eda_urgent.py`

# %% [markdown]
# ## 0. Setup

# %%
import warnings
warnings.filterwarnings("ignore")

import hashlib, os, re, string, yaml
from pathlib import Path
from collections import Counter
from datetime import date

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu, pointbiserialr, kruskal
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# -- NLTK resources (download once) -----------------------------------------
for resource in ["stopwords", "punkt", "punkt_tab", "rslp"]:
    try:
        nltk.data.find(f"tokenizers/{resource}" if resource.startswith("punkt") else f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

# -- Config (single source of truth for all paths and data version) ----------
ROOT        = Path(__file__).resolve().parent.parent
_cfg_path   = ROOT / "configs" / "config.yaml"
try:
    with open(_cfg_path) as _f:
        _cfg = yaml.safe_load(_f)
    DATA_VERSION = _cfg.get("data", {}).get("version", "unknown")
    _fig_cfg     = _cfg.get("eda", {})
    _DPI         = _fig_cfg.get("dpi", 120)
    _FIG_W       = _fig_cfg.get("fig_width", 12)
    _FIG_H       = _fig_cfg.get("fig_height", 6)
    _REPORTS_DIR = ROOT / _cfg.get("eda", {}).get("reports_dir", "docs")
    _FIGS_BASE   = ROOT / _cfg.get("eda", {}).get("figures_dir", "docs/figures")
except Exception as _e:
    print(f"WARNING: could not load config — using defaults: {_e}")
    DATA_VERSION = "unknown"
    _DPI, _FIG_W, _FIG_H = 120, 12, 6
    _REPORTS_DIR = ROOT / "docs"
    _FIGS_BASE   = ROOT / "docs" / "figures"

DATA_RAW = ROOT / _cfg.get("data", {}).get("raw_dir", "data/raw") / _cfg.get("data", {}).get("raw_filename", "todo_tasks.parquet")
FIG_DIR  = _FIGS_BASE / "urgent"
FIG_DIR.mkdir(parents=True, exist_ok=True)
_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# -- Versioned figure helper: saves <name>_<data_version>.png AND <name>.png --
def _savefig(name: str, fig=None) -> None:
    """Save figure to FIG_DIR as both versioned and canonical filenames."""
    _fig = fig or plt.gcf()
    stem = name.replace(".png", "")
    versioned = FIG_DIR / f"{stem}_{DATA_VERSION}.png"
    canonical = FIG_DIR / f"{stem}.png"
    for _p in [versioned, canonical]:
        _fig.savefig(_p, dpi=_DPI, bbox_inches="tight")
    print(f"Saved: {canonical.name}  (versioned: {versioned.name})")

# -- Plot style --------------------------------------------------------------
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
PALETTE      = {0: "#2196F3", 1: "#E53935"}    # blue = not urgent, red = urgent
PALETTE_STR  = {"0": "#2196F3", "1": "#E53935"} # string keys for seaborn categorical axis

print(f"Setup complete.  data_version={DATA_VERSION}")
print(f"Data path : {DATA_RAW}")
print(f"Figures   : {FIG_DIR}")

# %% [markdown]
# ## 1. Load & Initial Inspection

# %%
# Drop columns not available at creation time or not relevant
COLS_TO_DROP = [
    "id", "comments", "project_id", "updated_at",
    "register_timesheet", "completed_date", "important"
]

df_raw = pd.read_parquet(DATA_RAW)
print(f"Raw shape : {df_raw.shape}")
print(f"\nColumns   : {df_raw.columns.tolist()}")
print(f"\nDtypes:\n{df_raw.dtypes}")

df = df_raw.drop(columns=COLS_TO_DROP)
print(f"\nWorking shape after drop: {df.shape}")
print(f"Working columns: {df.columns.tolist()}")

# %%
print("\n--- Null Counts ---")
print(df.isnull().sum())

print("\n--- Sample (5 rows) ---")
pd.set_option("display.max_colwidth", 80)
print(df.sample(5, random_state=42).to_string())

# %%
# Sanity check: confirm urgent column exists
if "urgent" not in df.columns:
    raise ValueError("Column 'urgent' not found in dataset. Check raw parquet.")

print(f"\n'urgent' column present. Unique values: {df['urgent'].unique()}")
print(f"Null count in 'urgent': {df['urgent'].isnull().sum()}")

# %% [markdown]
# ## 2. Target Variable Distribution

# %%
target_counts = df["urgent"].value_counts().sort_index()
target_pct    = df["urgent"].value_counts(normalize=True).sort_index() * 100

print("=== Target: urgent ===")
for cls, cnt in target_counts.items():
    label = "urgent" if cls == 1 else "not urgent"
    print(f"  {cls} ({label}): {cnt:>5}  ({target_pct[cls]:.1f}%)")

if len(target_counts) == 2:
    majority = target_counts.max()
    minority = target_counts.min()
    imbalance_ratio = majority / minority
    print(f"\nImbalance ratio (majority/minority): {imbalance_ratio:.2f}:1")
else:
    print("\nWARNING: Unexpected number of classes in 'urgent'.")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Count bar
axes[0].bar(
    ["Not Urgent (0)", "Urgent (1)"],
    target_counts.values,
    color=[PALETTE[0], PALETTE[1]], edgecolor="white", linewidth=1.5
)
for i, v in enumerate(target_counts.values):
    axes[0].text(i, v + 30, str(v), ha="center", fontweight="bold")
axes[0].set_title("Class Distribution — Count")
axes[0].set_ylabel("Count")

# Percentage pie
axes[1].pie(
    target_counts.values,
    labels=[f"Not Urgent\n{target_pct[0]:.1f}%", f"Urgent\n{target_pct[1]:.1f}%"],
    colors=[PALETTE[0], PALETTE[1]],
    autopct="%1.1f%%", startangle=90,
    wedgeprops={"edgecolor": "white", "linewidth": 2}
)
axes[1].set_title("Class Distribution — Proportion")

plt.tight_layout()
_savefig("01_target_distribution")
plt.show()

# %% [markdown]
# ## 3. Feature Engineering
#
# **Only features available at task creation time.**
# - `days_until_due`  : due_date - created_at (planned lead time — known at creation)
# - `lead_time_days`  : alias for days_until_due (explicit naming)
# - `hour_created`    : hour of day task was created
# - `day_of_week_created` : weekday task was created (0=Mon…6=Sun)
# - `month_created`   : month task was created
# - `desc_char_len`   : character length of description
# - `desc_word_count` : word count of description
# - `project_category`: Personal vs Work (inferred from project_name at creation)
#
# **Excluded (not available at creation time)**:
# - `is_overdue`      : requires comparing due_date to today → only known after due date passes

# %%
# --- Date parsing -----------------------------------------------------------
for col in ["due_date", "created_at"]:
    df[col] = pd.to_datetime(df[col], errors="coerce")

# --- Temporal features (creation-time only) ---------------------------------
df["lead_time_days"]      = (df["due_date"] - df["created_at"]).dt.days
df["lead_time_hours"]     = (df["due_date"] - df["created_at"]).dt.total_seconds() / 3600
df["day_of_week_created"] = df["created_at"].dt.dayofweek   # 0=Mon … 6=Sun
df["month_created"]       = df["created_at"].dt.month
df["hour_created"]        = df["created_at"].dt.hour

# --- Lead time buckets (useful for urgency — tighter deadlines = more urgent) ---
# Negative = task created after due date (data quality issue)
df["lead_time_bucket"] = pd.cut(
    df["lead_time_days"],
    bins=[-np.inf, 0, 1, 3, 7, 14, 30, np.inf],
    labels=["negative/same_day", "1_day", "2-3_days", "4-7_days", "1-2_weeks", "2-4_weeks", ">4_weeks"]
)

# --- Text features ----------------------------------------------------------
df["description"] = df["description"].fillna("").str.strip()
df["desc_char_len"]   = df["description"].str.len()
df["desc_word_count"] = df["description"].str.split().str.len()

# --- Project category (Personal vs Work) ------------------------------------
df["project_category"] = df["project_name"].apply(
    lambda x: "Personal" if "personal" in str(x).lower() else "Work"
)

print("Engineered features added:")
new_cols = [
    "lead_time_days", "lead_time_hours", "lead_time_bucket",
    "day_of_week_created", "month_created", "hour_created",
    "desc_char_len", "desc_word_count", "project_category"
]
for c in new_cols:
    print(f"  {c}: {df[c].dtype}  nulls={df[c].isnull().sum()}")

print(f"\nFinal working shape: {df.shape}")

# %%
# Data quality: check for negative lead times (due_date < created_at)
neg_lead = df[df["lead_time_days"] < 0]
print(f"\nNegative lead time (due_date < created_at): {len(neg_lead)} rows")
if len(neg_lead) > 0:
    print(neg_lead[["description", "created_at", "due_date", "lead_time_days", "urgent"]].head(10).to_string())

# %%
# Summary statistics for key numeric feature
print("\n--- lead_time_days summary ---")
print(df["lead_time_days"].describe(percentiles=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]))

print("\n--- lead_time_days by urgent class ---")
for cls in [0, 1]:
    sub = df.loc[df["urgent"] == cls, "lead_time_days"].dropna()
    label = "Urgent" if cls == 1 else "Not Urgent"
    print(f"  {label}: median={sub.median():.1f}  mean={sub.mean():.1f}  std={sub.std():.1f}  n={len(sub)}")

# %% [markdown]
# ## 4. Categorical Features

# %% [markdown]
# ### 4.1 project_name vs urgent

# %%
n = len(df)

ct_project = pd.crosstab(df["project_name"], df["urgent"], margins=True, margins_name="Total")
ct_project.columns = ["Not Urgent", "Urgent", "Total"]
ct_project["Pct Urgent"] = (ct_project["Urgent"] / ct_project["Total"] * 100).round(1)
print("=== Contingency Table: project_name vs urgent ===")
print(ct_project.to_string())

chi2_proj, p_proj, dof_proj, _ = chi2_contingency(pd.crosstab(df["project_name"], df["urgent"]))
cramers_v_proj = np.sqrt(chi2_proj / (n * (min(pd.crosstab(df["project_name"], df["urgent"]).shape) - 1)))
print(f"\nChi2={chi2_proj:.2f}  p={p_proj:.4e}  dof={dof_proj}  Cramér's V={cramers_v_proj:.4f}")

# %%
proj_pct = df.groupby("project_name")["urgent"].value_counts(normalize=True).unstack().fillna(0) * 100
proj_pct.columns = ["Not Urgent", "Urgent"]
proj_pct = proj_pct.sort_values("Urgent", ascending=True)

fig, ax = plt.subplots(figsize=(10, 5))
proj_pct.plot(kind="barh", stacked=True, ax=ax,
              color=[PALETTE[0], PALETTE[1]], edgecolor="white")
ax.set_xlabel("Percentage (%)")
ax.set_title(f"Urgent Rate by Project  (Cramér's V={cramers_v_proj:.3f})")
ax.legend(loc="lower right")
ax.xaxis.set_major_formatter(mticker.PercentFormatter())
plt.tight_layout()
_savefig("02_project_vs_urgent")
plt.show()

# %% [markdown]
# ### 4.2 project_category (Personal vs Work) vs urgent

# %%
ct_cat = pd.crosstab(df["project_category"], df["urgent"], margins=True, margins_name="Total")
ct_cat.columns = ["Not Urgent", "Urgent", "Total"]
ct_cat["Pct Urgent"] = (ct_cat["Urgent"] / ct_cat["Total"] * 100).round(1)
print("=== Contingency Table: project_category vs urgent ===")
print(ct_cat.to_string())

chi2_cat, p_cat, dof_cat, _ = chi2_contingency(pd.crosstab(df["project_category"], df["urgent"]))
cramers_v_cat = np.sqrt(chi2_cat / (n * (min(pd.crosstab(df["project_category"], df["urgent"]).shape) - 1)))
print(f"\nChi2={chi2_cat:.2f}  p={p_cat:.4e}  dof={dof_cat}  Cramér's V={cramers_v_cat:.4f}")

# %%
fig, ax = plt.subplots(figsize=(6, 4))
ct_raw = pd.crosstab(df["project_category"], df["urgent"])
ct_raw.plot(kind="bar", ax=ax, color=[PALETTE[0], PALETTE[1]], edgecolor="white", rot=0)
ax.set_title(f"Personal vs Work — Urgent Count  (Cramér's V={cramers_v_cat:.3f})")
ax.set_ylabel("Count")
ax.legend(["Not Urgent", "Urgent"])
plt.tight_layout()
_savefig("03_category_vs_urgent")
plt.show()
print("Saved: 03_category_vs_urgent.png")

# %% [markdown]
# ### 4.3 project_code vs urgent

# %%
ct_code = pd.crosstab(df["project_code"], df["urgent"], margins=True, margins_name="Total")
ct_code.columns = ["Not Urgent", "Urgent", "Total"]
ct_code["Pct Urgent"] = (ct_code["Urgent"] / ct_code["Total"] * 100).round(1)
print("=== Contingency Table: project_code vs urgent ===")
print(ct_code.to_string())

chi2_code, p_code, dof_code, _ = chi2_contingency(pd.crosstab(df["project_code"], df["urgent"]))
cramers_v_code = np.sqrt(chi2_code / (n * (min(pd.crosstab(df["project_code"], df["urgent"]).shape) - 1)))
print(f"\nChi2={chi2_code:.2f}  p={p_code:.4e}  dof={dof_code}  Cramér's V={cramers_v_code:.4f}")

# %% [markdown]
# ## 5. Temporal Feature Analysis

# %% [markdown]
# ### 5.1 lead_time_days (due_date − created_at)
#
# **Urgency hypothesis**: tasks with short planned lead time are more likely to be urgent.
# A task due in 1 day is more urgent than one due in 30 days.

# %%
lead_0 = df.loc[df["urgent"] == 0, "lead_time_days"].dropna()
lead_1 = df.loc[df["urgent"] == 1, "lead_time_days"].dropna()

stat_u_lead, p_u_lead = mannwhitneyu(lead_0, lead_1, alternative="two-sided")
r_lead, p_r_lead = pointbiserialr(
    df.loc[df["lead_time_days"].notna(), "urgent"],
    df.loc[df["lead_time_days"].notna(), "lead_time_days"]
)

print("=== lead_time_days vs urgent ===")
print(f"  Not Urgent: median={lead_0.median():.1f}  mean={lead_0.mean():.1f}  n={len(lead_0)}")
print(f"  Urgent    : median={lead_1.median():.1f}  mean={lead_1.mean():.1f}  n={len(lead_1)}")
print(f"  Mann-Whitney U={stat_u_lead:.0f}  p={p_u_lead:.4e}")
print(f"  Point-biserial r={r_lead:.4f}  p={p_r_lead:.4e}")

# %%
clip_lo = df["lead_time_days"].quantile(0.01)
clip_hi = df["lead_time_days"].quantile(0.99)
plot_lead = df[df["lead_time_days"].between(clip_lo, clip_hi)].copy()
plot_lead["urgent_label"] = plot_lead["urgent"].map({0: "Not Urgent", 1: "Urgent"})

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

sns.boxplot(data=plot_lead, x="urgent", y="lead_time_days",
            palette=PALETTE_STR, ax=axes[0])
axes[0].set_xticklabels(["Not Urgent", "Urgent"])
axes[0].set_title("Lead Time (days) — Boxplot")
axes[0].set_ylabel("Days until due")

for cls, color in PALETTE.items():
    sub = plot_lead.loc[plot_lead["urgent"] == cls, "lead_time_days"].dropna()
    label = "Urgent" if cls == 1 else "Not Urgent"
    axes[1].hist(sub, bins=40, alpha=0.5, color=color, label=label, density=True)
axes[1].set_title("Lead Time — Distribution")
axes[1].set_xlabel("Days until due")
axes[1].set_ylabel("Density")
axes[1].legend()

plt.suptitle(f"lead_time_days by urgent class  (Mann-Whitney p={p_u_lead:.2e}, r={r_lead:.3f})",
             fontsize=11, y=1.01)
plt.tight_layout()
_savefig("04_lead_time_vs_urgent")
plt.show()
print("Saved: 04_lead_time_vs_urgent.png")

# %% [markdown]
# ### 5.2 lead_time_bucket vs urgent (categorical view)

# %%
ct_bucket = pd.crosstab(df["lead_time_bucket"], df["urgent"], margins=True, margins_name="Total")
ct_bucket.columns = ["Not Urgent", "Urgent", "Total"]
ct_bucket["Pct Urgent"] = (ct_bucket["Urgent"] / ct_bucket["Total"] * 100).round(1)
print("=== Contingency Table: lead_time_bucket vs urgent ===")
print(ct_bucket.to_string())

ct_bucket_raw = pd.crosstab(df["lead_time_bucket"], df["urgent"])
if ct_bucket_raw.shape[0] >= 2:
    chi2_bucket, p_bucket, dof_bucket, _ = chi2_contingency(ct_bucket_raw)
    cramers_v_bucket = np.sqrt(chi2_bucket / (n * (min(ct_bucket_raw.shape) - 1)))
    print(f"\nChi2={chi2_bucket:.2f}  p={p_bucket:.4e}  dof={dof_bucket}  Cramér's V={cramers_v_bucket:.4f}")
else:
    chi2_bucket, p_bucket, cramers_v_bucket = 0.0, 1.0, 0.0
    print("NOTE: lead_time_bucket has too few categories.")

# %%
bucket_pct = df.groupby("lead_time_bucket", observed=True)["urgent"].value_counts(normalize=True).unstack().fillna(0) * 100
bucket_pct.columns = ["Not Urgent", "Urgent"]

fig, ax = plt.subplots(figsize=(10, 4))
bucket_pct.plot(kind="bar", stacked=True, ax=ax,
                color=[PALETTE[0], PALETTE[1]], edgecolor="white", rot=20)
ax.set_title(f"Urgent Rate by Lead Time Bucket  (Cramér's V={cramers_v_bucket:.3f})")
ax.set_ylabel("Percentage (%)")
ax.set_xlabel("Lead Time Bucket")
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
ax.legend(loc="lower right")
plt.tight_layout()
_savefig("05_lead_time_bucket_vs_urgent")
plt.show()

# %% [markdown]
# ### 5.3 day_of_week_created vs urgent

# %%
day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
ct_dow = pd.crosstab(df["day_of_week_created"], df["urgent"])
ct_dow.index = [day_names[i] for i in ct_dow.index]

chi2_dow, p_dow, dof_dow, _ = chi2_contingency(ct_dow)
cramers_v_dow = np.sqrt(chi2_dow / (n * (min(ct_dow.shape) - 1)))
print(f"Day of Week — Chi2={chi2_dow:.2f}  p={p_dow:.4e}  Cramér's V={cramers_v_dow:.4f}")

ct_dow_pct = ct_dow.div(ct_dow.sum(axis=1), axis=0) * 100
ct_dow_pct.columns = ["Not Urgent", "Urgent"]

fig, ax = plt.subplots(figsize=(8, 4))
ct_dow_pct.plot(kind="bar", stacked=True, ax=ax,
                color=[PALETTE[0], PALETTE[1]], edgecolor="white", rot=0)
ax.set_title(f"Urgent Rate by Day of Week  (Cramér's V={cramers_v_dow:.3f})")
ax.set_ylabel("Percentage (%)")
ax.set_xlabel("Day of Week (task created)")
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
ax.legend(loc="lower right")
plt.tight_layout()
_savefig("06_dow_vs_urgent")
plt.show()

# %% [markdown]
# ### 5.4 month_created vs urgent

# %%
ct_month = pd.crosstab(df["month_created"], df["urgent"])
chi2_month, p_month, dof_month, _ = chi2_contingency(ct_month)
cramers_v_month = np.sqrt(chi2_month / (n * (min(ct_month.shape) - 1)))
print(f"Month — Chi2={chi2_month:.2f}  p={p_month:.4e}  Cramér's V={cramers_v_month:.4f}")

ct_month_pct = ct_month.div(ct_month.sum(axis=1), axis=0) * 100
ct_month_pct.columns = ["Not Urgent", "Urgent"]

fig, ax = plt.subplots(figsize=(10, 4))
ct_month_pct.plot(kind="bar", stacked=True, ax=ax,
                  color=[PALETTE[0], PALETTE[1]], edgecolor="white", rot=0)
ax.set_title(f"Urgent Rate by Month  (Cramér's V={cramers_v_month:.3f})")
ax.set_ylabel("Percentage (%)")
ax.set_xlabel("Month (task created)")
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
plt.tight_layout()
_savefig("07_month_vs_urgent")
plt.show()

# %% [markdown]
# ### 5.5 hour_created vs urgent

# %%
ct_hour = pd.crosstab(df["hour_created"], df["urgent"])
chi2_hour, p_hour, dof_hour, _ = chi2_contingency(ct_hour)
cramers_v_hour = np.sqrt(chi2_hour / (n * (min(ct_hour.shape) - 1)))
print(f"Hour — Chi2={chi2_hour:.2f}  p={p_hour:.4e}  Cramér's V={cramers_v_hour:.4f}")

ct_hour_pct = ct_hour.div(ct_hour.sum(axis=1), axis=0) * 100
ct_hour_pct.columns = ["Not Urgent", "Urgent"]

fig, ax = plt.subplots(figsize=(12, 4))
ct_hour_pct.plot(kind="bar", stacked=True, ax=ax,
                 color=[PALETTE[0], PALETTE[1]], edgecolor="white", rot=0)
ax.set_title(f"Urgent Rate by Hour of Day  (Cramér's V={cramers_v_hour:.3f})")
ax.set_ylabel("Percentage (%)")
ax.set_xlabel("Hour (task created)")
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
plt.tight_layout()
_savefig("08_hour_vs_urgent")
plt.show()

# %% [markdown]
# ### 5.6 lead_time_hours — fine-grained same-day urgency

# %%
# For tasks with lead_time_days <= 3, explore hours-level granularity
short_lead = df[df["lead_time_days"] <= 3].copy()
print(f"Tasks with lead_time_days <= 3: {len(short_lead)} ({len(short_lead)/n*100:.1f}% of total)")
print(f"  Urgent within short-lead: {short_lead['urgent'].mean()*100:.1f}%")

if len(short_lead) > 10 and short_lead["urgent"].nunique() == 2:
    r_lt_hrs, p_lt_hrs = pointbiserialr(short_lead["urgent"], short_lead["lead_time_hours"])
    stat_u_hrs, p_u_hrs = mannwhitneyu(
        short_lead.loc[short_lead["urgent"] == 0, "lead_time_hours"],
        short_lead.loc[short_lead["urgent"] == 1, "lead_time_hours"],
        alternative="two-sided"
    )
    print(f"\nWithin short-lead tasks:")
    print(f"  Point-biserial r (lead_time_hours vs urgent) = {r_lt_hrs:.4f}  p={p_lt_hrs:.4e}")
    print(f"  Mann-Whitney U={stat_u_hrs:.0f}  p={p_u_hrs:.4e}")

    fig, ax = plt.subplots(figsize=(8, 4))
    for cls, color in PALETTE.items():
        sub = short_lead.loc[short_lead["urgent"] == cls, "lead_time_hours"]
        label = "Urgent" if cls == 1 else "Not Urgent"
        ax.hist(sub, bins=30, alpha=0.5, color=color, label=label, density=True)
    ax.set_title("Lead Time (hours) — Short-lead tasks only (<= 3 days)")
    ax.set_xlabel("Hours until due")
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()
    _savefig("09_lead_time_hours_shortlead")
    plt.show()
else:
    r_lt_hrs, p_lt_hrs = 0.0, 1.0
    print("Insufficient data for short-lead hours analysis.")

# %% [markdown]
# ## 6. Text Feature Analysis (description)

# %%
PT_STOPWORDS = set(stopwords.words("portuguese"))

def clean_text(text: str) -> list[str]:
    """Lowercase, remove punctuation and digits, tokenize, remove stopwords."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text, language="portuguese")
    return [t for t in tokens if t not in PT_STOPWORDS and len(t) > 2]

df["tokens"]     = df["description"].apply(clean_text)
df["desc_clean"] = df["tokens"].apply(lambda t: " ".join(t))

print("Token sample (first 3 rows):")
for i, row in df.head(3).iterrows():
    print(f"  [{row['urgent']}] '{row['description']}' -> {row['tokens']}")

# %% [markdown]
# ### 6.1 Description length vs urgent

# %%
r_char, p_char = pointbiserialr(df["urgent"], df["desc_char_len"])
r_word, p_word = pointbiserialr(df["urgent"], df["desc_word_count"])

char_0 = df.loc[df["urgent"] == 0, "desc_char_len"]
char_1 = df.loc[df["urgent"] == 1, "desc_char_len"]
mw_char_stat, mw_char_p = mannwhitneyu(char_0, char_1, alternative="two-sided")

print("=== Description Length ===")
print(f"  Not Urgent: mean={char_0.mean():.1f} chars  median={char_0.median():.1f}")
print(f"  Urgent    : mean={char_1.mean():.1f} chars  median={char_1.median():.1f}")
print(f"  Point-biserial r={r_char:.4f}  p={p_char:.4e}")
print(f"  Mann-Whitney U={mw_char_stat:.0f}  p={mw_char_p:.4e}")
print(f"\n=== Word Count ===")
print(f"  Point-biserial r={r_word:.4f}  p={p_word:.4e}")

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

clip_hi_char = df["desc_char_len"].quantile(0.99)
plot_char = df[df["desc_char_len"] <= clip_hi_char]

sns.boxplot(data=plot_char, x="urgent", y="desc_char_len",
            palette=PALETTE_STR, ax=axes[0])
axes[0].set_xticklabels(["Not Urgent", "Urgent"])
axes[0].set_title(f"Char Length (r={r_char:.3f}, p={p_char:.2e})")
axes[0].set_ylabel("Characters")

sns.boxplot(data=df, x="urgent", y="desc_word_count",
            palette=PALETTE_STR, ax=axes[1])
axes[1].set_xticklabels(["Not Urgent", "Urgent"])
axes[1].set_title(f"Word Count (r={r_word:.3f}, p={p_word:.2e})")
axes[1].set_ylabel("Words")

plt.tight_layout()
_savefig("10_desc_length_vs_urgent")
plt.show()

# %% [markdown]
# ### 6.2 Top N-grams per class

# %%
def get_top_ngrams(texts: pd.Series, n: int, top_k: int = 20) -> pd.DataFrame:
    """Return top-k n-grams from a text series."""
    all_tokens = " ".join(texts).split()
    if n == 1:
        ngrams = all_tokens
    else:
        ngrams = [" ".join(all_tokens[i:i+n]) for i in range(len(all_tokens) - n + 1)]
    counts = Counter(ngrams)
    return pd.DataFrame(counts.most_common(top_k), columns=["ngram", "count"])

for n, label in [(1, "Unigrams"), (2, "Bigrams")]:
    texts_0 = df.loc[df["urgent"] == 0, "desc_clean"]
    texts_1 = df.loc[df["urgent"] == 1, "desc_clean"]

    top_0 = get_top_ngrams(texts_0, n)
    top_1 = get_top_ngrams(texts_1, n)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, top, cls_label, color in zip(
        axes,
        [top_0, top_1],
        ["Not Urgent", "Urgent"],
        [PALETTE[0], PALETTE[1]]
    ):
        ax.barh(top["ngram"][::-1], top["count"][::-1], color=color)
        ax.set_title(f"Top {label} — {cls_label}")
        ax.set_xlabel("Frequency")

    plt.suptitle(f"Top {label} by Urgent Class", fontsize=13)
    plt.tight_layout()
    _savefig(f"11_top_{label.lower()}_by_class")
    plt.show()

# %% [markdown]
# ### 6.3 TF-IDF top discriminating features

# %%
tfidf = TfidfVectorizer(max_features=200, ngram_range=(1, 2))
X_tfidf = tfidf.fit_transform(df["desc_clean"])
tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=tfidf.get_feature_names_out())

pb_results = []
for col in tfidf_df.columns:
    r, p = pointbiserialr(df["urgent"], tfidf_df[col])
    pb_results.append({"feature": col, "r": r, "p": p, "abs_r": abs(r)})

pb_df = pd.DataFrame(pb_results).sort_values("abs_r", ascending=False)
print("=== Top 20 TF-IDF features by |Point-Biserial r| (vs urgent) ===")
print(pb_df.head(20).to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 6))
top20 = pb_df.head(20).sort_values("r")
colors = [PALETTE[1] if r > 0 else PALETTE[0] for r in top20["r"]]
ax.barh(top20["feature"], top20["r"], color=colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("TF-IDF Features — Point-Biserial r with 'urgent'")
ax.set_xlabel("Point-Biserial r  (positive = correlated with urgent=1)")
plt.tight_layout()
_savefig("12_tfidf_point_biserial")
plt.show()

# %% [markdown]
# ### 6.4 Word Clouds per class

# %%
try:
    from wordcloud import WordCloud

    for cls in [0, 1]:
        label = "Urgent" if cls == 1 else "NotUrgent"
        text = " ".join(df.loc[df["urgent"] == cls, "desc_clean"])
        wc = WordCloud(
            width=800, height=400,
            background_color="white",
            colormap="Blues" if cls == 0 else "Reds",
            max_words=100
        ).generate(text)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"Word Cloud — {label.replace('Not', 'Not ')}")
        plt.tight_layout()
        _savefig(f"13_wordcloud_{label.lower()}")
        plt.show()

except ImportError:
    print("wordcloud not available — skipping word clouds")

# %% [markdown]
# ## 7. Numeric Feature Correlations

# %%
numeric_feats = ["lead_time_days", "lead_time_hours", "desc_char_len",
                 "desc_word_count", "hour_created"]

print("=== Point-Biserial Correlation with 'urgent' ===")
pb_numeric = []
for feat in numeric_feats:
    vals = df[[feat, "urgent"]].dropna()
    r, p = pointbiserialr(vals["urgent"], vals[feat])
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    pb_numeric.append({"feature": feat, "r": r, "p": p, "sig": sig})
    print(f"  {feat:<25} r={r:+.4f}  p={p:.4e}  {sig}")

pb_numeric_df = pd.DataFrame(pb_numeric).sort_values("r", key=abs, ascending=False)

# %%
plot_df = df[numeric_feats + ["urgent"]].dropna()
plot_df["urgent_label"] = plot_df["urgent"].map({0: "Not Urgent", 1: "Urgent"})
fig = sns.pairplot(
    plot_df.sample(min(1000, len(plot_df)), random_state=42),
    hue="urgent_label",
    palette={"Not Urgent": "#2196F3", "Urgent": "#E53935"},
    vars=numeric_feats,
    plot_kws={"alpha": 0.4, "s": 20},
    diag_kind="kde"
)
fig.figure.suptitle("Pairplot — Numeric Features vs Urgent", y=1.01)
_savefig("14_pairplot_numeric", fig=fig.figure)
plt.show()

# %% [markdown]
# ## 8. Multivariate Analysis
#
# Cross-relationships between urgency and combinations of features.
# Helps detect interaction effects that a univariate analysis misses.

# %% [markdown]
# ### 8.1 lead_time_days × project_category × urgent

# %%
print("=== Median lead_time_days by project_category x urgent ===")
pivot = df.groupby(["project_category", "urgent"])["lead_time_days"].median().unstack()
pivot.columns = [("Not Urgent" if c == 0 else "Urgent") for c in pivot.columns]
# Fill NaN with 0 for categories with no urgent tasks (avoids plot error)
pivot = pivot.fillna(0)
print(pivot.to_string())

fig, ax = plt.subplots(figsize=(7, 4))
pivot.plot(kind="bar", ax=ax, color=[PALETTE[0], PALETTE[1]], edgecolor="white", rot=0)
ax.set_title("Median Lead Time by Category × Urgency")
ax.set_ylabel("Median days until due")
ax.legend(["Not Urgent", "Urgent"])
plt.tight_layout()
_savefig("15_lead_time_by_category_urgent")
plt.show()

# %% [markdown]
# ### 8.2 hour_created × lead_time_bucket heatmap — urgent rate

# %%
df_heat = df.dropna(subset=["lead_time_bucket"]).copy()
pivot_heat = df_heat.groupby(["hour_created", "lead_time_bucket"], observed=True)["urgent"].mean() * 100
pivot_heat = pivot_heat.unstack(level="lead_time_bucket")

if pivot_heat.shape[0] > 0 and pivot_heat.shape[1] > 0:
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(
        pivot_heat,
        annot=True, fmt=".0f", cmap="Reds",
        linewidths=0.5, ax=ax,
        cbar_kws={"label": "% Urgent"}
    )
    ax.set_title("Urgent Rate (%) — Hour Created × Lead Time Bucket")
    ax.set_ylabel("Hour of Day Created")
    ax.set_xlabel("Lead Time Bucket")
    plt.tight_layout()
    _savefig("16_heatmap_hour_leadtime_urgent")
    plt.show()

# %% [markdown]
# ### 8.3 day_of_week × lead_time_bucket — urgent rate heatmap

# %%
day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
df_heat2 = df.dropna(subset=["lead_time_bucket"]).copy()
df_heat2["dow_name"] = df_heat2["day_of_week_created"].map(dict(enumerate(day_names)))

pivot_heat2 = df_heat2.groupby(["dow_name", "lead_time_bucket"], observed=True)["urgent"].mean() * 100
pivot_heat2 = pivot_heat2.unstack(level="lead_time_bucket")
pivot_heat2 = pivot_heat2.reindex(day_names)

if pivot_heat2.shape[0] > 0:
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.heatmap(
        pivot_heat2,
        annot=True, fmt=".0f", cmap="Reds",
        linewidths=0.5, ax=ax,
        cbar_kws={"label": "% Urgent"}
    )
    ax.set_title("Urgent Rate (%) — Day of Week × Lead Time Bucket")
    ax.set_ylabel("Day of Week")
    ax.set_xlabel("Lead Time Bucket")
    plt.tight_layout()
    _savefig("17_heatmap_dow_leadtime_urgent")
    plt.show()

# %% [markdown]
# ## 9. Consolidated Statistical Summary

# %%
summary_rows = [
    # Categoricals
    {"feature": "project_name",       "type": "categorical", "test": "Chi2 + Cramér's V",
     "statistic": chi2_proj,    "p_value": p_proj,       "effect_size": cramers_v_proj},
    {"feature": "project_code",       "type": "categorical", "test": "Chi2 + Cramér's V",
     "statistic": chi2_code,    "p_value": p_code,       "effect_size": cramers_v_code},
    {"feature": "project_category",   "type": "categorical", "test": "Chi2 + Cramér's V",
     "statistic": chi2_cat,     "p_value": p_cat,        "effect_size": cramers_v_cat},
    {"feature": "lead_time_bucket",   "type": "categorical", "test": "Chi2 + Cramér's V",
     "statistic": chi2_bucket,  "p_value": p_bucket,     "effect_size": cramers_v_bucket},
    {"feature": "day_of_week",        "type": "categorical", "test": "Chi2 + Cramér's V",
     "statistic": chi2_dow,     "p_value": p_dow,        "effect_size": cramers_v_dow},
    {"feature": "month_created",      "type": "categorical", "test": "Chi2 + Cramér's V",
     "statistic": chi2_month,   "p_value": p_month,      "effect_size": cramers_v_month},
    {"feature": "hour_created",       "type": "categorical", "test": "Chi2 + Cramér's V",
     "statistic": chi2_hour,    "p_value": p_hour,       "effect_size": cramers_v_hour},
    # Continuous
    {"feature": "lead_time_days",     "type": "continuous",  "test": "Mann-Whitney U + r",
     "statistic": stat_u_lead,  "p_value": p_u_lead,     "effect_size": abs(r_lead)},
    {"feature": "desc_char_len",      "type": "continuous",  "test": "Point-biserial r",
     "statistic": r_char,       "p_value": p_char,       "effect_size": abs(r_char)},
    {"feature": "desc_word_count",    "type": "continuous",  "test": "Point-biserial r",
     "statistic": r_word,       "p_value": p_word,       "effect_size": abs(r_word)},
]

summary_df = pd.DataFrame(summary_rows)
summary_df["significant"] = summary_df["p_value"] < 0.05
summary_df["significance"] = summary_df["p_value"].apply(
    lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
)
summary_df = summary_df.sort_values("p_value")

print("=== CONSOLIDATED STATISTICAL SUMMARY ===")
print(summary_df.to_string(index=False))

# %%
effect_plot = summary_df.dropna(subset=["effect_size"]).copy()
effect_plot["abs_effect"] = effect_plot["effect_size"].abs()
effect_plot = effect_plot.sort_values("abs_effect", ascending=True)

fig, ax = plt.subplots(figsize=(9, 6))
colors = [PALETTE[1] if sig else "#AAAAAA" for sig in effect_plot["significant"]]
ax.barh(effect_plot["feature"], effect_plot["abs_effect"], color=colors)
ax.axvline(0.1, color="orange", linestyle="--", linewidth=1, label="Small effect (0.1)")
ax.axvline(0.3, color="darkred", linestyle="--", linewidth=1, label="Medium effect (0.3)")
ax.set_title("Effect Size per Feature (Cramér's V / |r|)  — Urgent target")
ax.set_xlabel("Effect Size")
ax.legend()
plt.tight_layout()
_savefig("18_effect_sizes")
plt.show()

# %% [markdown]
# ## 10. Label Consistency Check
#
# Assess whether `urgent` labels are self-consistent:
# - Do identical or near-identical tasks get the same label?
# - Is there evidence the label is noisy/subjective?
# This informs whether a classifier is viable or whether a rule-based approach is safer.

# %%
# Check urgent rate within each project — high variance may indicate subjectivity
proj_urgent_rate = df.groupby("project_name")["urgent"].agg(["mean", "count"]).reset_index()
proj_urgent_rate.columns = ["project_name", "urgent_rate", "count"]
proj_urgent_rate = proj_urgent_rate.sort_values("urgent_rate", ascending=False)

print("=== Urgent rate by project (sorted) ===")
print(proj_urgent_rate.to_string(index=False))

rate_std = proj_urgent_rate["urgent_rate"].std()
print(f"\nStd of urgent rate across projects: {rate_std:.3f}")
print("  High std (>0.2) suggests project-level variation — project is a useful feature.")
print("  If most projects cluster near the global rate, project may not add value.")

# %%
# Check lead_time_days distribution for potentially mislabeled cases:
# short lead = expected urgent; long lead = expected not urgent.
# Flag contradictions.
print("\n=== Potential label contradictions ===")
# Urgent=0 but very short lead time (< 1 day)
contradictions_0 = df[(df["urgent"] == 0) & (df["lead_time_days"] <= 1)]
print(f"  Not Urgent with lead_time <= 1 day : {len(contradictions_0)}")

# Urgent=1 but very long lead time (> 30 days)
contradictions_1 = df[(df["urgent"] == 1) & (df["lead_time_days"] > 30)]
print(f"  Urgent with lead_time > 30 days   : {len(contradictions_1)}")

total_contradictions = len(contradictions_0) + len(contradictions_1)
contradiction_rate = total_contradictions / len(df) * 100
print(f"  Total contradiction candidates    : {total_contradictions} ({contradiction_rate:.1f}% of data)")
print("Note: these are 'contradictions' only if urgency ~ deadline proximity.")
print("      A task can be urgent for non-temporal reasons (e.g. blocking dependency).")

# %% [markdown]
# ## 11. Model Strategy Recommendation
#
# Based on EDA findings, decide between:
# A) Supervised classifier (LightGBM / Logistic Regression)
# B) Rule-based engine (threshold on lead_time_days + project)
# C) Clustering (only if labels are too noisy)

# %%
print("=" * 70)
print("MODEL STRATEGY ANALYSIS")
print("=" * 70)

print(
    "Key questions answered by EDA:\n\n"
    "1. IS `urgent` PREDICTABLE FROM CREATION-TIME FEATURES?\n"
    "   - lead_time_days is the primary signal (hypothesis: shorter = more urgent).\n"
    "   - hour_created, day_of_week, project may add signal.\n"
    "   - See consolidated p-values above.\n\n"
    "2. IS THE LABEL CONSISTENT ENOUGH FOR SUPERVISED LEARNING?\n"
    "   - Check contradiction rate above. If < 20% contradictions -> classifier viable.\n"
    "   - If > 30% contradictions -> consider rule-based or semi-supervised approach.\n\n"
    "3. RULE-BASED VIABILITY:\n"
    "   - If lead_time_days alone achieves high separation (AUC > 0.75 in a simple\n"
    "     threshold test), a rule-based approach (lead_time <= N days = urgent) may be\n"
    "     interpretable and sufficient for production.\n"
    "   - Rules are always preferable when they generalize as well as a model.\n\n"
    "4. CLUSTERING (Eisenhower quadrant):\n"
    "   - Clustering is NOT recommended for production labeling (unsupervised != label).\n"
    "   - Could be used for EDA discovery only (e.g. k-means on lead_time + importance).\n"
    "   - Do NOT use clustering as a substitute for supervised learning here.\n"
)

# Quick threshold test: what fraction of urgent tasks have lead_time_days <= 7?
for threshold in [1, 3, 7, 14, 30]:
    n_urgent_under = ((df["lead_time_days"] <= threshold) & (df["urgent"] == 1)).sum()
    n_urgent_total = (df["urgent"] == 1).sum()
    n_total_under  = (df["lead_time_days"] <= threshold).sum()
    precision = n_urgent_under / n_total_under * 100 if n_total_under > 0 else 0
    recall    = n_urgent_under / n_urgent_total * 100 if n_urgent_total > 0 else 0
    print(f"  Rule: lead_time <= {threshold:>2} days -> Precision={precision:.1f}%  Recall={recall:.1f}%")

# %% [markdown]
# ## 12. Save Processed Data

# %%
_processed_dir = ROOT / _cfg.get("data", {}).get("processed_dir", "data/processed")
PROCESSED_PATH = _processed_dir / "eda_urgent_features.parquet"
PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)

save_cols = [
    "description", "desc_clean", "project_name", "project_code",
    "project_category", "due_date", "created_at",
    "lead_time_days", "lead_time_hours", "lead_time_bucket",
    "day_of_week_created", "month_created", "hour_created",
    "desc_char_len", "desc_word_count",
    "urgent"
]
df[save_cols].to_parquet(PROCESSED_PATH, index=False)
print(f"Processed data saved to: {PROCESSED_PATH}")
print(f"Shape: {df[save_cols].shape}")

# %% [markdown]
# ## 12b. Generate Versioned EDA Report

# %%
# -- Compute parquet SHA-256 for traceability --------------------------------
_sha256 = "unknown"
try:
    _h = hashlib.sha256()
    with open(PROCESSED_PATH, "rb") as _f:
        for _chunk in iter(lambda: _f.read(1 << 20), b""):
            _h.update(_chunk)
    _sha256 = _h.hexdigest()
except Exception as _e:
    print(f"Warning: could not compute SHA-256: {_e}")

# -- Build report content ----------------------------------------------------
_n_total   = len(df)
_n_urgent  = int(df["urgent"].sum())
_n_not     = _n_total - _n_urgent
_pct_urg   = _n_urgent / _n_total * 100
_imbalance = _n_not / _n_urgent if _n_urgent > 0 else float("inf")

_report_lines = [
    f"# EDA Report — Model 2: Urgent Classifier",
    f"",
    f"**Data version**: `{DATA_VERSION}`  ",
    f"**Report date**: {date.today().isoformat()}  ",
    f"**Processed data SHA-256**: `{_sha256}`  ",
    f"**Source file**: `data/processed/eda_urgent_features.parquet`  ",
    f"",
    f"---",
    f"",
    f"## Dataset",
    f"",
    f"| Statistic | Value |",
    f"|---|---|",
    f"| Total rows | {_n_total:,} |",
    f"| Urgent (1) | {_n_urgent:,} ({_pct_urg:.1f}%) |",
    f"| Not Urgent (0) | {_n_not:,} ({100-_pct_urg:.1f}%) |",
    f"| Imbalance ratio (majority:minority) | {_imbalance:.1f}:1 |",
    f"| Features saved | {len(save_cols) - 1} |",
    f"",
    f"---",
    f"",
    f"## Key EDA Findings (Effect Sizes)",
    f"",
]

try:
    _report_lines.append("```")
    _report_lines.append(summary_df[["feature", "test", "effect_size", "significance"]].to_string(index=False))
    _report_lines.append("```")
    _report_lines.append("")
except Exception as _e:
    _report_lines.append(f"_(Summary table not available: {_e})_")
    _report_lines.append("")

_report_lines += [
    f"---",
    f"",
    f"## Top TF-IDF Text Features",
    f"",
]
try:
    _report_lines.append("```")
    _report_lines.append(pb_df.head(10)[["feature", "r", "p"]].to_string(index=False))
    _report_lines.append("```")
    _report_lines.append("")
except Exception as _e:
    _report_lines.append(f"_(TF-IDF analysis not available: {_e})_")
    _report_lines.append("")

_report_lines += [
    f"---",
    f"",
    f"## Numeric Point-Biserial Correlations",
    f"",
]
try:
    _pb_num = pd.DataFrame(pb_numeric).sort_values("r", key=abs, ascending=False)
    _report_lines.append("```")
    _report_lines.append(_pb_num.to_string(index=False))
    _report_lines.append("```")
    _report_lines.append("")
except Exception as _e:
    _report_lines.append(f"_(Numeric correlations not available: {_e})_")
    _report_lines.append("")

_report_lines += [
    f"---",
    f"",
    f"## Lead Time Threshold Rules",
    f"",
    f"| Threshold (days) | Precision (%) | Recall (%) |",
    f"|---|---|---|",
]
for _t in [1, 3, 7, 14, 30]:
    _nu = ((df["lead_time_days"] <= _t) & (df["urgent"] == 1)).sum()
    _nt = (df["lead_time_days"] <= _t).sum()
    _prec = _nu / _nt * 100 if _nt > 0 else 0
    _rec  = _nu / _n_urgent * 100 if _n_urgent > 0 else 0
    _report_lines.append(f"| <= {_t} | {_prec:.1f} | {_rec:.1f} |")
_report_lines.append("")

_report_lines += [
    f"---",
    f"",
    f"## Figures",
    f"",
    f"See `docs/figures/urgent/` for all {len(list(FIG_DIR.glob('*.png')))} generated figures.",
    f"",
]

_report_content = "\n".join(_report_lines)

# -- Write versioned copy (never overwrites a previous version) --------------
_versioned_name = f"eda_urgent_report_{DATA_VERSION}.md"
_versioned_path = _REPORTS_DIR / _versioned_name
_canonical_path = _REPORTS_DIR / "eda_urgent_report.md"

for _path in [_versioned_path, _canonical_path]:
    _path.write_text(_report_content, encoding="utf-8")
    print(f"EDA report written to: {_path}")

print(f"\nData version : {DATA_VERSION}")
print(f"SHA-256      : {_sha256[:16]}...{_sha256[-8:]}")
print(f"Rows saved   : {_n_total:,}")
print(f"Imbalance    : {_imbalance:.1f}:1  (urgent={_pct_urg:.1f}%)")

# %% [markdown]
# ## 13. Feature Ranking Summary

# %%
print("\n" + "=" * 60)
print("FEATURE RANKING SUMMARY (by predictive strength — urgent)")
print("=" * 60)
print(summary_df[["feature", "test", "effect_size", "significance"]].to_string(index=False))

print("\n--- TF-IDF Top 10 Text Features ---")
print(pb_df.head(10)[["feature", "r", "p"]].to_string(index=False))

print("\n--- Numeric Point-Biserial (Top) ---")
pb_numeric_sorted = pd.DataFrame(pb_numeric).sort_values("r", key=abs, ascending=False)
print(pb_numeric_sorted.to_string(index=False))

print("\n--- Lead Time Threshold Rules ---")
for threshold in [1, 3, 7, 14]:
    n_urgent_under = ((df["lead_time_days"] <= threshold) & (df["urgent"] == 1)).sum()
    n_urgent_total = (df["urgent"] == 1).sum()
    n_total_under  = (df["lead_time_days"] <= threshold).sum()
    precision = n_urgent_under / n_total_under * 100 if n_total_under > 0 else 0
    recall    = n_urgent_under / n_urgent_total * 100 if n_urgent_total > 0 else 0
    print(f"  lead_time <= {threshold:>2} days: Precision={precision:.1f}%  Recall={recall:.1f}%")
