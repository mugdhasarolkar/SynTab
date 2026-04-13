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