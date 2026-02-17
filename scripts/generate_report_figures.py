from __future__ import annotations

import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = os.path.dirname(os.path.dirname(__file__))
RAW_CSV = os.path.join(ROOT, "data", "raw", "data.csv")
OUT_DIR = os.path.join(ROOT, "reports", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

def compute_stats(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    stats = df[cols].describe().T
    stats = stats.rename(columns={"50%": "median"})
    stats["skew"] = df[cols].skew()
    outlier_pct = []
    for c in cols:
        q1 = df[c].quantile(0.25)
        q3 = df[c].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        out = df[(df[c] < lower) | (df[c] > upper)]
        outlier_pct.append(100 * len(out) / len(df))
    stats["outlier_%"] = outlier_pct
    return stats

def save_boxplots(df: pd.DataFrame, col: str, out_dir: str) -> None:
    fname = os.path.join(out_dir, f"boxplot_{col.lower()}.png")
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[col].dropna())
    plt.title(f"Boxplot: {col} (raw)")
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()

    # Also save a log1p boxplot to show distribution with heavy tails
    fname_log = os.path.join(out_dir, f"boxplot_{col.lower()}_log1p.png")
    plt.figure(figsize=(6,4))
    sns.boxplot(x=np.log1p(df[col].clip(lower=0).fillna(0)))
    plt.title(f"Boxplot: log1p({col})")
    plt.tight_layout()
    plt.savefig(fname_log)
    plt.close()

def main() -> None:
    df = pd.read_csv(RAW_CSV)
    # Ensure numeric
    cols = ["Amount", "Value"]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    stats = compute_stats(df, cols)
    stats_csv = os.path.join(OUT_DIR, "raw_summary_stats.csv")
    stats.to_csv(stats_csv)
    print("Saved stats to:", stats_csv)

    # Save boxplots
    for c in cols:
        save_boxplots(df, c, OUT_DIR)
        print("Saved boxplots for:", c)

if __name__ == '__main__':
    main()
