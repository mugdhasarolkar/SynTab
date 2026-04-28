import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from preprocessing.preprocess import load_data, preprocess
from models.train import train_vae
from models.generate import generate_full_dataset
from models.vtae import run_synthesizer
from evaluation.evaluate import ks_evaluation
from evaluation.plot import plot_ks_summary, plot_top_k_distributions

st.set_page_config(
    page_title="Syntab",
    page_icon="📊",
    layout="wide"
)

st.markdown(
    "<h1 style='text-align: center;'>Syntab</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center; font-size:18px;'>Synthetic Tabular Data Generation Platform</p>",
    unsafe_allow_html=True
)

st.markdown("---")

st.subheader("Upload Dataset")

uploaded_file = st.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx"]
)

st.info("Only tabular datasets (CSV/Excel) are supported.")

if uploaded_file is not None:
    df = load_data(uploaded_file)
    real_df = df.copy()

    st.subheader("Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    k = st.slider("Number of features to visualize", 1, 10, 3)

    if st.button("Generate Synthetic Data"):
        with st.spinner("Processing... This may take a few minutes."):
            df_processed, scaler, encoder, num_features, num_imputer, cat_imputer = preprocess(df)
            X = df_processed.to_numpy()

            train_vae(X, scaler, encoder, num_features)

            n_samples = X.shape[0]
            generated_df = generate_full_dataset(n_samples)

            num_cols = joblib.load("outputs/saved_models/num_cols.pkl")
            ks_df, avg_ks, similarity_score = ks_evaluation(real_df, generated_df, num_cols)

            st.subheader("Evaluation Results")
            st.write(f"Average KS Score: **{avg_ks:.4f}**")

            if avg_ks < 0.4:
                st.success("VAE output accepted")
                final_df = generated_df
            else:
                st.warning("VAE not sufficient → Switching to TVAE")
                final_df = run_synthesizer(real_df)
                ks_df, avg_ks, similarity_score = ks_evaluation(real_df, final_df, num_cols)
                st.write(f"TVAE KS Score: **{avg_ks:.4f}**")

        st.markdown("---")
        st.subheader("Visual Analysis")

        # First Figure: KS Summary (Full Width)
        st.markdown("### 1. Kolmogorov-Smirnov Score per Feature")
        fig1 = plot_ks_summary(ks_df)
        fig1.set_size_inches(14, 7) # Increased width and height
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig1, use_container_width=True)

        st.markdown("<br><br>", unsafe_allow_html=True) # Adding space between figures
        st.markdown("---")

        # Second Figure: Top K Distributions (Full Width, Vertical Stack)
        st.markdown(f"### 2. Top {k} Feature Distributions (Real vs Synthetic)")
        fig2 = plot_top_k_distributions(real_df, final_df, ks_df, k=k)
        
        # Adjust height dynamically based on k to prevent squishing
        fig2.set_size_inches(14, 6 * k) 
        fig2.tight_layout(pad=6.0) # Increases padding between subplots
        st.pyplot(fig2, use_container_width=True)

        st.markdown("---")
        csv = final_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Synthetic Dataset",
            data=csv,
            file_name="synthetic_data.csv",
            mime="text/csv"
        )

st.markdown("---")
st.markdown(
    "<p style='text-align: center;'>Built with VAE-based Synthetic Data Generation</p>",
    unsafe_allow_html=True
)