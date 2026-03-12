# %% [markdown]
# # EDA — Model 1: Important / Not Important
#
# **Goal**: Understand which features best predict whether a task is `important=1`.
# **Target variable**: `important` (binary: 0 = not important, 1 = important)
#
# Run interactively with `# %%` cells in VS Code (Python Interactive) or
# execute end-to-end with `python notebooks/eda_important.py`

# %% [markdown]
# ## 0. Setup

# %%
import warnings
warnings.filterwarnings("ignore")

import os, re, string
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu, pointbiserialr
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

# -- Paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw" / "todo_tasks.parquet"
FIG_DIR  = ROOT / "docs" / "figures" / "important"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# -- Plot style --------------------------------------------------------------
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
PALETTE      = {0: "#4C72B0", 1: "#DD8452"}    # blue = not important, orange = important
PALETTE_STR  = {"0": "#4C72B0", "1": "#DD8452"} # string keys for seaborn categorical axis

print("Setup complete.")
print(f"Data path : {DATA_RAW}")
print(f"Figures   : {FIG_DIR}")

# %% [markdown]
# ## 1. Load & Initial Inspection

# %%
COLS_TO_DROP = ["id", "comments", "project_id", "updated_at",
                "register_timesheet", "completed_date", "urgent"]

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

# %% [markdown]
# ## 2. Target Variable Distribution

# %%
target_counts = df["important"].value_counts().sort_index()
target_pct    = df["important"].value_counts(normalize=True).sort_index() * 100

print("=== Target: important ===")
for cls, cnt in target_counts.items():
    label = "important" if cls == 1 else "not important"
    print(f"  {cls} ({label}): {cnt:>5}  ({target_pct[cls]:.1f}%)")

imbalance_ratio = target_counts[1] / target_counts[0]
print(f"\nImbalance ratio (majority/minority): {imbalance_ratio:.2f}:1")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Count bar
axes[0].bar(["Not important (0)", "Important (1)"],
            target_counts.values,
            color=[PALETTE[0], PALETTE[1]], edgecolor="white", linewidth=1.5)
for i, v in enumerate(target_counts.values):
    axes[0].text(i, v + 30, str(v), ha="center", fontweight="bold")
axes[0].set_title("Class Distribution — Count")
axes[0].set_ylabel("Count")

# Percentage pie
axes[1].pie(target_counts.values,
            labels=[f"Not important\n{target_pct[0]:.1f}%",
                    f"Important\n{target_pct[1]:.1f}%"],
            colors=[PALETTE[0], PALETTE[1]],
            autopct="%1.1f%%", startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 2})
axes[1].set_title("Class Distribution — Proportion")

plt.tight_layout()
plt.savefig(FIG_DIR / "01_target_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: 01_target_distribution.png")

# %% [markdown]
# ## 3. Feature Engineering

# %%
# --- Date parsing -----------------------------------------------------------
for col in ["due_date", "created_at"]:
    df[col] = pd.to_datetime(df[col], errors="coerce")

# --- Temporal features ------------------------------------------------------
REFERENCE_DATE = pd.Timestamp("2026-03-11")   # today at time of analysis

df["days_until_due"]    = (df["due_date"] - df["created_at"]).dt.days
df["days_since_created"] = (REFERENCE_DATE - df["created_at"]).dt.days
df["is_overdue"]         = ((df["due_date"] < REFERENCE_DATE) & df["due_date"].notna()).astype(int)
df["day_of_week_created"] = df["created_at"].dt.dayofweek          # 0=Mon … 6=Sun
df["month_created"]       = df["created_at"].dt.month
df["hour_created"]        = df["created_at"].dt.hour
df["has_due_date"]        = df["due_date"].notna().astype(int)

# --- Text features ----------------------------------------------------------
df["description"] = df["description"].fillna("").str.strip()
df["desc_char_len"]  = df["description"].str.len()
df["desc_word_count"] = df["description"].str.split().str.len()

# --- Project category (Personal vs Work) ------------------------------------
df["project_category"] = df["project_name"].apply(
    lambda x: "Personal" if "personal" in str(x).lower() else "Work"
)

print("Engineered features added:")
new_cols = ["days_until_due", "days_since_created", "is_overdue",
            "day_of_week_created", "month_created", "hour_created",
            "has_due_date", "desc_char_len", "desc_word_count", "project_category"]
for c in new_cols:
    print(f"  {c}: {df[c].dtype}  nulls={df[c].isnull().sum()}")

print(f"\nFinal working shape: {df.shape}")

# %% [markdown]
# ## 4. Categorical Features

# %% [markdown]
# ### 4.1 project_name vs important

# %%
ct_project = pd.crosstab(df["project_name"], df["important"],
                          margins=True, margins_name="Total")
ct_project.columns = ["Not Important", "Important", "Total"]
ct_project["Pct Important"] = (ct_project["Important"] / ct_project["Total"] * 100).round(1)
print("=== Contingency Table: project_name vs important ===")
print(ct_project.to_string())

# Chi-squared
chi2_proj, p_proj, dof_proj, _ = chi2_contingency(
    pd.crosstab(df["project_name"], df["important"])
)
print(f"\nChi2={chi2_proj:.2f}  p={p_proj:.4e}  dof={dof_proj}")

# Cramér's V
n = len(df)
cramers_v_proj = np.sqrt(chi2_proj / (n * (min(pd.crosstab(df["project_name"], df["important"]).shape) - 1)))
print(f"Cramér's V = {cramers_v_proj:.4f}")

# %%
# Stacked bar — % important per project
proj_pct = df.groupby("project_name")["important"].value_counts(normalize=True).unstack().fillna(0) * 100
proj_pct.columns = ["Not Important", "Important"]
proj_pct = proj_pct.sort_values("Important", ascending=True)

fig, ax = plt.subplots(figsize=(10, 5))
proj_pct.plot(kind="barh", stacked=True, ax=ax,
              color=[PALETTE[0], PALETTE[1]], edgecolor="white")
ax.set_xlabel("Percentage (%)")
ax.set_title("Important Rate by Project")
ax.legend(loc="lower right")
ax.xaxis.set_major_formatter(mticker.PercentFormatter())
plt.tight_layout()
plt.savefig(FIG_DIR / "02_project_vs_important.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: 02_project_vs_important.png")

# %% [markdown]
# ### 4.2 project_category (Personal vs Work) vs important

# %%
ct_cat = pd.crosstab(df["project_category"], df["important"],
                     margins=True, margins_name="Total")
ct_cat.columns = ["Not Important", "Important", "Total"]
ct_cat["Pct Important"] = (ct_cat["Important"] / ct_cat["Total"] * 100).round(1)
print("=== Contingency Table: project_category vs important ===")
print(ct_cat.to_string())

chi2_cat, p_cat, dof_cat, _ = chi2_contingency(
    pd.crosstab(df["project_category"], df["important"])
)
cramers_v_cat = np.sqrt(chi2_cat / (n * (min(pd.crosstab(df["project_category"], df["important"]).shape) - 1)))
print(f"\nChi2={chi2_cat:.2f}  p={p_cat:.4e}  dof={dof_cat}  Cramér's V={cramers_v_cat:.4f}")

# %%
fig, ax = plt.subplots(figsize=(6, 4))
ct_raw = pd.crosstab(df["project_category"], df["important"])
ct_raw.plot(kind="bar", ax=ax, color=[PALETTE[0], PALETTE[1]],
            edgecolor="white", rot=0)
ax.set_title("Personal vs Work — Important Count")
ax.set_ylabel("Count")
ax.legend(["Not Important", "Important"])
plt.tight_layout()
plt.savefig(FIG_DIR / "03_category_vs_important.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: 03_category_vs_important.png")

# %% [markdown]
# ### 4.3 project_code vs important

# %%
ct_code = pd.crosstab(df["project_code"], df["important"],
                      margins=True, margins_name="Total")
ct_code.columns = ["Not Important", "Important", "Total"]
ct_code["Pct Important"] = (ct_code["Important"] / ct_code["Total"] * 100).round(1)
print("=== Contingency Table: project_code vs important ===")
print(ct_code.to_string())

chi2_code, p_code, dof_code, _ = chi2_contingency(
    pd.crosstab(df["project_code"], df["important"])
)
cramers_v_code = np.sqrt(chi2_code / (n * (min(pd.crosstab(df["project_code"], df["important"]).shape) - 1)))
print(f"\nChi2={chi2_code:.2f}  p={p_code:.4e}  dof={dof_code}  Cramér's V={cramers_v_code:.4f}")

# %% [markdown]
# ## 5. Temporal Feature Analysis

# %% [markdown]
# ### 5.1 days_until_due (lead time)

# %%
# Drop nulls for this analysis
days_0 = df.loc[df["important"] == 0, "days_until_due"].dropna()
days_1 = df.loc[df["important"] == 1, "days_until_due"].dropna()

stat_u, p_u = mannwhitneyu(days_0, days_1, alternative="two-sided")
print("=== days_until_due ===")
print(f"  Not important: median={days_0.median():.1f}  mean={days_0.mean():.1f}  n={len(days_0)}")
print(f"  Important    : median={days_1.median():.1f}  mean={days_1.mean():.1f}  n={len(days_1)}")
print(f"  Mann-Whitney U={stat_u:.0f}  p={p_u:.4e}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Clip extreme outliers for readability (keep 1–99 percentile)
clip_lo = df["days_until_due"].quantile(0.01)
clip_hi = df["days_until_due"].quantile(0.99)
plot_data = df[df["days_until_due"].between(clip_lo, clip_hi)].copy()

# Boxplot
sns.boxplot(data=plot_data, x="important", y="days_until_due",
            palette=PALETTE_STR, ax=axes[0])
axes[0].set_xticklabels(["Not Important", "Important"])
axes[0].set_title("Lead Time (days_until_due) — Boxplot")
axes[0].set_ylabel("Days")

# KDE
for cls, color in PALETTE.items():
    sub = plot_data.loc[plot_data["important"] == cls, "days_until_due"].dropna()
    label = "Important" if cls == 1 else "Not Important"
    axes[1].hist(sub, bins=40, alpha=0.5, color=color, label=label, density=True)
axes[1].set_title("Lead Time — Distribution")
axes[1].set_xlabel("Days until due")
axes[1].set_ylabel("Density")
axes[1].legend()

plt.suptitle(f"days_until_due by important class  (Mann-Whitney p={p_u:.2e})",
             fontsize=11, y=1.01)
plt.tight_layout()
plt.savefig(FIG_DIR / "04_lead_time_vs_important.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: 04_lead_time_vs_important.png")

# %% [markdown]
# ### 5.2 has_due_date vs important

# %%
ct_due = pd.crosstab(df["has_due_date"], df["important"],
                     margins=True, margins_name="Total")
ct_due.columns = ["Not Important", "Important", "Total"]
ct_due["Pct Important"] = (ct_due["Important"] / ct_due["Total"] * 100).round(1)
# Rename index dynamically (might be only 1 value if all rows have due dates)
idx_map = {0: "No Due Date", 1: "Has Due Date"}
ct_due.index = [idx_map.get(i, "Total") if i != "Total" else "Total" for i in ct_due.index]
print("=== Contingency Table: has_due_date vs important ===")
print(ct_due.to_string())

if df["has_due_date"].nunique() < 2:
    chi2_due, p_due, dof_due, cramers_v_due = 0.0, 1.0, 0, 0.0
    print("\nNOTE: has_due_date is constant (all tasks have a due date) — Chi2 not applicable")
else:
    chi2_due, p_due, dof_due, _ = chi2_contingency(
        pd.crosstab(df["has_due_date"], df["important"])
    )
    cramers_v_due = np.sqrt(chi2_due / (n * (min(pd.crosstab(df["has_due_date"], df["important"]).shape) - 1)))
    print(f"\nChi2={chi2_due:.2f}  p={p_due:.4e}  dof={dof_due}  Cramér's V={cramers_v_due:.4f}")

# %% [markdown]
# ### 5.3 is_overdue vs important

# %%
ct_over = pd.crosstab(df["is_overdue"], df["important"],
                      margins=True, margins_name="Total")
ct_over.columns = ["Not Important", "Important", "Total"]
ct_over["Pct Important"] = (ct_over["Important"] / ct_over["Total"] * 100).round(1)
idx_map_over = {0: "Not Overdue", 1: "Overdue"}
ct_over.index = [idx_map_over.get(i, "Total") if i != "Total" else "Total" for i in ct_over.index]
print("=== Contingency Table: is_overdue vs important ===")
print(ct_over.to_string())

chi2_over, p_over, dof_over, _ = chi2_contingency(
    pd.crosstab(df["is_overdue"], df["important"])
)
cramers_v_over = np.sqrt(chi2_over / (n * (min(pd.crosstab(df["is_overdue"], df["important"]).shape) - 1)))
print(f"\nChi2={chi2_over:.2f}  p={p_over:.4e}  dof={dof_over}  Cramér's V={cramers_v_over:.4f}")

# %% [markdown]
# ### 5.4 day_of_week_created vs important

# %%
day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
ct_dow = pd.crosstab(df["day_of_week_created"], df["important"])
ct_dow.index = [day_names[i] for i in ct_dow.index]

chi2_dow, p_dow, dof_dow, _ = chi2_contingency(ct_dow)
cramers_v_dow = np.sqrt(chi2_dow / (n * (min(ct_dow.shape) - 1)))
print(f"Day of Week — Chi2={chi2_dow:.2f}  p={p_dow:.4e}  Cramér's V={cramers_v_dow:.4f}")

ct_dow_pct = ct_dow.div(ct_dow.sum(axis=1), axis=0) * 100
ct_dow_pct.columns = ["Not Important", "Important"]

fig, ax = plt.subplots(figsize=(8, 4))
ct_dow_pct.plot(kind="bar", stacked=True, ax=ax,
                color=[PALETTE[0], PALETTE[1]], edgecolor="white", rot=0)
ax.set_title(f"Important Rate by Day of Week  (Cramér's V={cramers_v_dow:.3f})")
ax.set_ylabel("Percentage (%)")
ax.set_xlabel("Day of Week (task created)")
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(FIG_DIR / "05_dow_vs_important.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: 05_dow_vs_important.png")

# %% [markdown]
# ### 5.5 month_created vs important

# %%
ct_month = pd.crosstab(df["month_created"], df["important"])
chi2_month, p_month, dof_month, _ = chi2_contingency(ct_month)
cramers_v_month = np.sqrt(chi2_month / (n * (min(ct_month.shape) - 1)))
print(f"Month — Chi2={chi2_month:.2f}  p={p_month:.4e}  Cramér's V={cramers_v_month:.4f}")

ct_month_pct = ct_month.div(ct_month.sum(axis=1), axis=0) * 100
ct_month_pct.columns = ["Not Important", "Important"]

fig, ax = plt.subplots(figsize=(10, 4))
ct_month_pct.plot(kind="bar", stacked=True, ax=ax,
                  color=[PALETTE[0], PALETTE[1]], edgecolor="white", rot=0)
ax.set_title(f"Important Rate by Month  (Cramér's V={cramers_v_month:.3f})")
ax.set_ylabel("Percentage (%)")
ax.set_xlabel("Month (task created)")
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
plt.tight_layout()
plt.savefig(FIG_DIR / "06_month_vs_important.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: 06_month_vs_important.png")

# %% [markdown]
# ### 5.6 hour_created vs important

# %%
ct_hour = pd.crosstab(df["hour_created"], df["important"])
chi2_hour, p_hour, dof_hour, _ = chi2_contingency(ct_hour)
cramers_v_hour = np.sqrt(chi2_hour / (n * (min(ct_hour.shape) - 1)))
print(f"Hour — Chi2={chi2_hour:.2f}  p={p_hour:.4e}  Cramér's V={cramers_v_hour:.4f}")

ct_hour_pct = ct_hour.div(ct_hour.sum(axis=1), axis=0) * 100
ct_hour_pct.columns = ["Not Important", "Important"]

fig, ax = plt.subplots(figsize=(12, 4))
ct_hour_pct.plot(kind="bar", stacked=True, ax=ax,
                 color=[PALETTE[0], PALETTE[1]], edgecolor="white", rot=0)
ax.set_title(f"Important Rate by Hour of Day  (Cramér's V={cramers_v_hour:.3f})")
ax.set_ylabel("Percentage (%)")
ax.set_xlabel("Hour (task created)")
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
plt.tight_layout()
plt.savefig(FIG_DIR / "07_hour_vs_important.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: 07_hour_vs_important.png")

# %% [markdown]
# ## 6. Text Feature Analysis (description)

# %%
# Portuguese stopwords
PT_STOPWORDS = set(stopwords.words("portuguese"))

def clean_text(text: str) -> list[str]:
    """Lowercase, remove punctuation and digits, tokenize, remove stopwords."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text, language="portuguese")
    return [t for t in tokens if t not in PT_STOPWORDS and len(t) > 2]

df["tokens"] = df["description"].apply(clean_text)
df["desc_clean"] = df["tokens"].apply(lambda t: " ".join(t))

print("Token sample (first 3 rows):")
for i, row in df.head(3).iterrows():
    print(f"  [{row['important']}] '{row['description']}' -> {row['tokens']}")

# %% [markdown]
# ### 6.1 Description length vs important

# %%
r_char, p_char = pointbiserialr(df["important"], df["desc_char_len"])
r_word, p_word = pointbiserialr(df["important"], df["desc_word_count"])

char_0 = df.loc[df["important"] == 0, "desc_char_len"]
char_1 = df.loc[df["important"] == 1, "desc_char_len"]
mw_char_stat, mw_char_p = mannwhitneyu(char_0, char_1, alternative="two-sided")

print("=== Description Length ===")
print(f"  Not important: mean={char_0.mean():.1f} chars  median={char_0.median():.1f}")
print(f"  Important    : mean={char_1.mean():.1f} chars  median={char_1.median():.1f}")
print(f"  Point-biserial r={r_char:.4f}  p={p_char:.4e}")
print(f"  Mann-Whitney U={mw_char_stat:.0f}  p={mw_char_p:.4e}")
print(f"\n=== Word Count ===")
print(f"  Point-biserial r={r_word:.4f}  p={p_word:.4e}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

clip_hi_char = df["desc_char_len"].quantile(0.99)
plot_char = df[df["desc_char_len"] <= clip_hi_char]

sns.boxplot(data=plot_char, x="important", y="desc_char_len",
            palette=PALETTE_STR, ax=axes[0])
axes[0].set_xticklabels(["Not Important", "Important"])
axes[0].set_title(f"Char Length (r={r_char:.3f}, p={p_char:.2e})")
axes[0].set_ylabel("Characters")

sns.boxplot(data=df, x="important", y="desc_word_count",
            palette=PALETTE_STR, ax=axes[1])
axes[1].set_xticklabels(["Not Important", "Important"])
axes[1].set_title(f"Word Count (r={r_word:.3f}, p={p_word:.2e})")
axes[1].set_ylabel("Words")

plt.tight_layout()
plt.savefig(FIG_DIR / "08_desc_length_vs_important.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: 08_desc_length_vs_important.png")

# %% [markdown]
# ### 6.2 Top N-grams per class

# %%
from collections import Counter

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
    texts_0 = df.loc[df["important"] == 0, "desc_clean"]
    texts_1 = df.loc[df["important"] == 1, "desc_clean"]

    top_0 = get_top_ngrams(texts_0, n)
    top_1 = get_top_ngrams(texts_1, n)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, top, cls_label, color in zip(
        axes,
        [top_0, top_1],
        ["Not Important", "Important"],
        [PALETTE[0], PALETTE[1]]
    ):
        ax.barh(top["ngram"][::-1], top["count"][::-1], color=color)
        ax.set_title(f"Top {label} — {cls_label}")
        ax.set_xlabel("Frequency")

    plt.suptitle(f"Top {label} by Class", fontsize=13)
    plt.tight_layout()
    fname = f"09_top_{label.lower()}_by_class.png"
    plt.savefig(FIG_DIR / fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {fname}")

# %% [markdown]
# ### 6.3 TF-IDF top discriminating features

# %%
tfidf = TfidfVectorizer(max_features=200, ngram_range=(1, 2))
X_tfidf = tfidf.fit_transform(df["desc_clean"])
tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=tfidf.get_feature_names_out())

# Point-biserial for each TF-IDF feature vs important
pb_results = []
for col in tfidf_df.columns:
    r, p = pointbiserialr(df["important"], tfidf_df[col])
    pb_results.append({"feature": col, "r": r, "p": p, "abs_r": abs(r)})

pb_df = pd.DataFrame(pb_results).sort_values("abs_r", ascending=False)
print("=== Top 20 TF-IDF features by |Point-Biserial r| ===")
print(pb_df.head(20).to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 6))
top20 = pb_df.head(20).sort_values("r")
colors = [PALETTE[1] if r > 0 else PALETTE[0] for r in top20["r"]]
ax.barh(top20["feature"], top20["r"], color=colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("TF-IDF Features — Point-Biserial r with 'important'")
ax.set_xlabel("Point-Biserial r  (positive = correlated with important=1)")
plt.tight_layout()
plt.savefig(FIG_DIR / "10_tfidf_point_biserial.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: 10_tfidf_point_biserial.png")

# %% [markdown]
# ### 6.4 Word Clouds per class

# %%
try:
    from wordcloud import WordCloud

    for cls in [0, 1]:
        label = "Important" if cls == 1 else "NotImportant"
        text = " ".join(df.loc[df["important"] == cls, "desc_clean"])
        wc = WordCloud(
            width=800, height=400,
            background_color="white",
            colormap="Blues" if cls == 0 else "Oranges",
            max_words=100
        ).generate(text)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"Word Cloud — {label.replace('Not', 'Not ').strip()}")
        plt.tight_layout()
        fname = f"11_wordcloud_{label.lower()}.png"
        plt.savefig(FIG_DIR / fname, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"Saved: {fname}")

except ImportError:
    print("wordcloud not available — skipping word clouds")

# %% [markdown]
# ## 7. Numeric Feature Correlations

# %%
numeric_feats = ["days_until_due", "days_since_created", "desc_char_len",
                 "desc_word_count", "hour_created"]

print("=== Point-Biserial Correlation with 'important' ===")
pb_numeric = []
for feat in numeric_feats:
    vals = df[[feat, "important"]].dropna()
    r, p = pointbiserialr(vals["important"], vals[feat])
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    pb_numeric.append({"feature": feat, "r": r, "p": p, "sig": sig})
    print(f"  {feat:<25} r={r:+.4f}  p={p:.4e}  {sig}")

pb_numeric_df = pd.DataFrame(pb_numeric).sort_values("r", key=abs, ascending=False)

# %%
# Pairplot of numeric engineered features coloured by important
plot_df = df[numeric_feats + ["important"]].dropna()
plot_df["important_label"] = plot_df["important"].map({0: "Not Important", 1: "Important"})
fig = sns.pairplot(
    plot_df.sample(min(1000, len(plot_df)), random_state=42),
    hue="important_label",
    palette={"Not Important": "#4C72B0", "Important": "#DD8452"},
    vars=numeric_feats,
    plot_kws={"alpha": 0.4, "s": 20},
    diag_kind="kde"
)
fig.figure.suptitle("Pairplot — Numeric Features vs Important", y=1.01)
plt.savefig(FIG_DIR / "12_pairplot_numeric.png", dpi=130, bbox_inches="tight")
plt.show()
print("Saved: 12_pairplot_numeric.png")

# %% [markdown]
# ## 8. Binary Flags Summary

# %%
binary_flags = ["has_due_date", "is_overdue"]

print("=== Binary Flags vs important ===")
flag_summary = []
for flag in binary_flags:
    ct_flag = pd.crosstab(df[flag], df["important"])
    if ct_flag.shape[0] < 2:
        print(f"  {flag:<20} CONSTANT — only one value present, skipping")
        flag_summary.append({"feature": flag, "chi2": 0.0, "p": 1.0, "cramers_v": 0.0})
        continue
    chi2, p, dof, _ = chi2_contingency(ct_flag)
    cv = np.sqrt(chi2 / (n * (min(ct_flag.shape) - 1)))
    flag_summary.append({"feature": flag, "chi2": chi2, "p": p, "cramers_v": cv})
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    print(f"  {flag:<20} Chi2={chi2:.2f}  p={p:.4e}  Cramér's V={cv:.4f}  {sig}")

# %% [markdown]
# ## 9. Consolidated Statistical Summary

# %%
summary_rows = [
    # Categoricals
    {"feature": "project_name",      "type": "categorical", "test": "Chi2 + Cramér's V",
     "statistic": chi2_proj,  "p_value": p_proj,   "effect_size": cramers_v_proj},
    {"feature": "project_code",      "type": "categorical", "test": "Chi2 + Cramér's V",
     "statistic": chi2_code,  "p_value": p_code,   "effect_size": cramers_v_code},
    {"feature": "project_category",  "type": "categorical", "test": "Chi2 + Cramér's V",
     "statistic": chi2_cat,   "p_value": p_cat,    "effect_size": cramers_v_cat},
    {"feature": "has_due_date",      "type": "binary",      "test": "Chi2 + Cramér's V",
     "statistic": chi2_due,   "p_value": p_due,    "effect_size": cramers_v_due},
    {"feature": "is_overdue",        "type": "binary",      "test": "Chi2 + Cramér's V",
     "statistic": chi2_over,  "p_value": p_over,   "effect_size": cramers_v_over},
    {"feature": "day_of_week",       "type": "categorical", "test": "Chi2 + Cramér's V",
     "statistic": chi2_dow,   "p_value": p_dow,    "effect_size": cramers_v_dow},
    {"feature": "month_created",     "type": "categorical", "test": "Chi2 + Cramér's V",
     "statistic": chi2_month, "p_value": p_month,  "effect_size": cramers_v_month},
    {"feature": "hour_created",      "type": "categorical", "test": "Chi2 + Cramér's V",
     "statistic": chi2_hour,  "p_value": p_hour,   "effect_size": cramers_v_hour},
    # Continuous (Mann-Whitney)
    {"feature": "days_until_due",    "type": "continuous",  "test": "Mann-Whitney U",
     "statistic": stat_u,     "p_value": p_u,      "effect_size": None},
    {"feature": "desc_char_len",     "type": "continuous",  "test": "Point-biserial r",
     "statistic": r_char,     "p_value": p_char,   "effect_size": r_char},
    {"feature": "desc_word_count",   "type": "continuous",  "test": "Point-biserial r",
     "statistic": r_word,     "p_value": p_word,   "effect_size": r_word},
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
# Visualize effect sizes
effect_plot = summary_df.dropna(subset=["effect_size"]).copy()
effect_plot["abs_effect"] = effect_plot["effect_size"].abs()
effect_plot = effect_plot.sort_values("abs_effect", ascending=True)

fig, ax = plt.subplots(figsize=(9, 6))
colors = [PALETTE[1] if sig else "#AAAAAA" for sig in effect_plot["significant"]]
ax.barh(effect_plot["feature"], effect_plot["abs_effect"], color=colors)
ax.axvline(0.1, color="orange", linestyle="--", linewidth=1, label="Small effect (0.1)")
ax.axvline(0.3, color="red",    linestyle="--", linewidth=1, label="Medium effect (0.3)")
ax.set_title("Effect Size per Feature (Cramér's V / |r|)")
ax.set_xlabel("Effect Size")
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "13_effect_sizes.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: 13_effect_sizes.png")

# %% [markdown]
# ## 10. Save Processed Data

# %%
PROCESSED_PATH = ROOT / "data" / "processed" / "eda_important_features.parquet"
PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)

save_cols = [
    "description", "desc_clean", "project_name", "project_code",
    "project_category", "due_date", "created_at",
    "days_until_due", "days_since_created", "is_overdue", "has_due_date",
    "day_of_week_created", "month_created", "hour_created",
    "desc_char_len", "desc_word_count",
    "important"
]
df[save_cols].to_parquet(PROCESSED_PATH, index=False)
print(f"Processed data saved to: {PROCESSED_PATH}")
print(f"Shape: {df[save_cols].shape}")

# %% [markdown]
# ## 11. Feature Ranking Summary

# %%
print("\n" + "="*60)
print("FEATURE RANKING SUMMARY (by predictive strength)")
print("="*60)
print(summary_df[["feature", "test", "effect_size", "significance"]].to_string(index=False))

print("\n--- TF-IDF Top 10 Text Features ---")
print(pb_df.head(10)[["feature", "r", "p"]].to_string(index=False))

print("\n--- Numeric Point-Biserial (Top) ---")
pb_numeric_sorted = pd.DataFrame(pb_numeric).sort_values("r", key=abs, ascending=False)
print(pb_numeric_sorted.to_string(index=False))
