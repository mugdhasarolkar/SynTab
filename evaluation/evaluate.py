import pandas as pd
from scipy.stats import ks_2samp

def ks_evaluation(real_df, synth_df, num_cols):
    ks_results = []

    for col in num_cols:
        if col in real_df.columns and col in synth_df.columns:

            stat, p_value = ks_2samp(
                real_df[col].dropna(),
                synth_df[col].dropna()
            )

            ks_results.append({
                "column": col,
                "ks_stat": stat,
                "p_value": p_value
            })

    ks_df = pd.DataFrame(ks_results)
    avg_ks = ks_df["ks_stat"].mean() if len(ks_df) > 0 else 0
    similarity_score = 1 - avg_ks

    return ks_df, avg_ks, similarity_score

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances


def memorization_check(X_real, X_gen):
    real_clean = X_real.fillna(0).astype(str)
    gen_clean = X_gen.fillna(0).astype(str)

    real_set = set(map(tuple, real_clean.values))
    gen_set = set(map(tuple, gen_clean.values))

    exact_matches = len(real_set.intersection(gen_set))
    exact_memorization = (exact_matches / len(gen_clean)) * 100

    return {
        "exact_memorization": float(exact_memorization),
        
    }
from scipy.stats import chi2_contingency


# Cramér’s V
def cramers_v(x, y):
    table = pd.crosstab(x, y)

    if table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0

    chi2 = chi2_contingency(table)[0]
    n = table.to_numpy().sum()

    r, k = table.shape
    return np.sqrt(chi2 / (n * (min(k - 1, r - 1) + 1e-9)))


import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


# -----------------------------
# Cramér’s V (categorical)
# -----------------------------
def cramers_v(x, y):
    table = pd.crosstab(x, y)

    if table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0

    chi2 = chi2_contingency(table)[0]
    n = table.to_numpy().sum()

    r, k = table.shape
    return np.sqrt(chi2 / (n * (min(r - 1, k - 1) + 1e-9)))


# -----------------------------
# MAIN FUNCTION
# -----------------------------
def correlation_compute(X_real, X_gen):
    drop_cols = [c for c in X_real.columns if "id" in c.lower()]

    X_real = X_real.drop(columns=drop_cols, errors="ignore")
    X_gen = X_gen.drop(columns=drop_cols, errors="ignore")

    X_gen = X_gen[X_real.columns]

    scores = []

    # =============================
    # 1. NUMERIC RELATIONSHIPS
    # =============================
    num_cols = X_real.select_dtypes(include=[np.number]).columns

    if len(num_cols) > 1:

        real_num = X_real[num_cols].corr()
        gen_num = X_gen[num_cols].corr()

        gen_num = gen_num.reindex_like(real_num)

        diff = (real_num - gen_num).abs()

        mask = ~np.eye(diff.shape[0], dtype=bool)

        num_score = 1 - np.nanmean(diff.values[mask])
        scores.append(num_score)


    # =============================
    # 2. CATEGORICAL RELATIONSHIPS
    # =============================
    cat_cols = X_real.select_dtypes(include=['object', 'category', 'bool']).columns

    if len(cat_cols) > 1:

        def cat_matrix(df):
            mat = pd.DataFrame(index=cat_cols, columns=cat_cols, dtype=float)

            for c1 in cat_cols:
                for c2 in cat_cols:
                    mat.loc[c1, c2] = cramers_v(df[c1], df[c2])

            return mat

        real_cat = cat_matrix(X_real)
        gen_cat = cat_matrix(X_gen)

        gen_cat = gen_cat.reindex_like(real_cat)

        diff = (real_cat - gen_cat).abs()

        mask = ~np.eye(diff.shape[0], dtype=bool)

        cat_score = 1 - np.nanmean(diff.values[mask])
        scores.append(cat_score)


    # =============================
    # FINAL SCORE
    # =============================
    if len(scores) == 0:
        return {"relationship_score": 0.0}

    return {
        "relationship_score": float(np.mean(scores))
    }