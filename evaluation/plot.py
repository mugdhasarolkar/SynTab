import matplotlib.pyplot as plt


def plot_ks_summary(ks_df, title="KS Score per Feature"):
    ks_df = ks_df.sort_values("ks_stat", ascending=False)

    fig, ax = plt.subplots(figsize=(6, 3))

    ax.bar(ks_df["column"], ks_df["ks_stat"])
    ax.set_xticklabels(ks_df["column"], rotation=45)
    ax.set_ylabel("KS Statistic")
    ax.set_title(title)

    plt.tight_layout()

    return fig


def get_top_k_features(ks_df, k=3):
    return ks_df.sort_values("ks_stat", ascending=False)["column"].head(k).tolist()


def plot_top_k_distributions(real_df, synth_df, ks_df, k=3, model_name="Synthetic"):
    top_cols = get_top_k_features(ks_df, k)

    fig, axes = plt.subplots(len(top_cols), 1, figsize=(5, 3 * len(top_cols)))

    # If only one plot, make it iterable
    if len(top_cols) == 1:
        axes = [axes]

    for i, col in enumerate(top_cols):
        axes[i].hist(real_df[col], bins=40, alpha=0.5, label="Real", density=True)
        axes[i].hist(synth_df[col], bins=40, alpha=0.5, label=model_name, density=True)

        axes[i].set_title(f"Feature: {col}")
        axes[i].legend()

    plt.tight_layout()

    return fig