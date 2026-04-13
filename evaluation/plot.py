import matplotlib.pyplot as plt

def plot_ks_summary(ks_df, title="KS Score per Feature"):
    ks_df = ks_df.sort_values("ks_stat", ascending=False)

    plt.figure(figsize=(10,5))
    plt.bar(ks_df["column"], ks_df["ks_stat"])
    plt.xticks(rotation=45)
    plt.ylabel("KS Statistic")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def get_top_k_features(ks_df, k=5):
    return ks_df.sort_values("ks_stat", ascending=False)["column"].head(k).tolist()


def plot_top_k_distributions(real_df, synth_df, ks_df, k=5, model_name="Synthetic"):
    top_cols = get_top_k_features(ks_df, k)

    for col in top_cols:
        plt.figure(figsize=(7,4))

        plt.hist(real_df[col], bins=40, alpha=0.5, label="Real", density=True)
        plt.hist(synth_df[col], bins=40, alpha=0.5, label=model_name, density=True)

        plt.title(f"Top Feature - {col}")
        plt.legend()
        plt.tight_layout()
        plt.show()