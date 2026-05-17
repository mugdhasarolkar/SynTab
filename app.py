import streamlit as st
import pandas as pd
from preprocessing.preprocess import load_data, preprocess
from models.train import train_vae
from models.generate import generate_full_dataset
from models.tvae import run_synthesizer
from evaluation.evaluate import ks_evaluation, memorization_check, correlation_compute
from evaluation.plot import plot_ks_summary, plot_top_k_distributions, get_top_k_features
import base64
from pathlib import Path
LOGO_PATH = Path("app/assets/logo.jpeg")

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SynTab",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)
# ─── GLOBAL CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
/* ===== Native container scroll styling (clean, non-hideous) ===== */
.block-container {
    padding: 1rem 1.5rem !important;
    box-sizing: border-box;
}
[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 12px !important;
    background: transparent !important;
}
[data-testid="stVerticalBlockBorderWrapper"]::-webkit-scrollbar { width: 8px; height: 8px; }
[data-testid="stVerticalBlockBorderWrapper"]::-webkit-scrollbar-track {
    background: transparent;
    border-radius: 10px;
}
[data-testid="stVerticalBlockBorderWrapper"]::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #cbd5e1 0%, #94a3b8 100%);
    border-radius: 999px;
    border: 2px solid transparent;
    background-clip: content-box;
}
[data-testid="stVerticalBlockBorderWrapper"]::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(180deg, #94a3b8 0%, #64748b 100%);
    background-clip: content-box;
}
[data-testid="stVerticalBlockBorderWrapper"] {
    scrollbar-width: thin;
    scrollbar-color: #94a3b8 transparent;
    scrollbar-gutter: stable;
}
/* ===== END native scroll styling ===== */
@keyframes fadeSlideDown  { from { opacity:0; transform:translateY(-10px); } to { opacity:1; transform:translateY(0); } }
@keyframes fadeSlideUp    { from { opacity:0; transform:translateY(14px);  } to { opacity:1; transform:translateY(0); } }
@keyframes fadeSlideLeft  { from { opacity:0; transform:translateX(-16px); } to { opacity:1; transform:translateX(0); } }
@keyframes fadeSlideRight { from { opacity:0; transform:translateX(16px);  } to { opacity:1; transform:translateX(0); } }
@keyframes popIn          { from { opacity:0; transform:scale(0.88);       } to { opacity:1; transform:scale(1);    } }
@keyframes fillBar        { from { width:0 !important; } to { } }
.left-pane  { animation: fadeSlideLeft  0.5s cubic-bezier(0.4,0,0.2,1) both; }
.right-pane { animation: fadeSlideRight 0.5s cubic-bezier(0.4,0,0.2,1) 0.15s both; }
.logo-block  { display:flex; align-items:center; gap:12px; margin-bottom:6px; animation:fadeSlideDown 0.4s ease 0.1s both; }
/* logo image (left of title, same row) */
.logo-img {
    width: 64px;
    height: 64px;
    border-radius: 12px;   /* optional: less “coin” if you prefer a square logo */
    object-fit: cover;
    flex-shrink: 0;
    display: block;
}
.logo-title  { font-size:22px; font-weight:600; color:#0f172a; margin:0; line-height:1.2; }
.logo-sub    { font-size:12px; color:#64748b; margin:0; }
.tagline { border-left:3px solid #3b82f6; padding:6px 0 6px 12px; font-size:13px; color:#475569; margin:12px 0; line-height:1.6; border-radius:0; animation:fadeSlideDown 0.4s ease 0.2s both; }
.check-row     { display:flex; align-items:center; gap:8px; font-size:12px; color:#475569; margin-bottom:5px; animation:fadeSlideDown 0.4s ease both; }
.check-row .ck { color:#22c55e; font-size:14px; }
.sec-label { font-size:10px; font-weight:600; text-transform:uppercase; letter-spacing:1.2px; color:#94a3b8; margin:14px 0 6px; }
.stFileUploader > div { border:1.5px dashed #cbd5e1 !important; border-radius:10px !important; transition:border-color 0.2s, background 0.2s !important; }
.stFileUploader > div:hover { border-color:#3b82f6 !important; background:#eff6ff !important; }
.stAlert { border-radius:8px !important; font-size:12px !important; }
.stButton > button {
    width:100% !important; background:#1d4ed8 !important; color:#fff !important;
    border:none !important; border-radius:8px !important; font-size:13px !important;
    font-weight:500 !important; padding:10px 0 !important;
    transition:opacity 0.15s, transform 0.1s !important;
    font-family:'DM Sans',sans-serif !important;
}
.stButton > button:hover  { opacity:0.88 !important; }
.stButton > button:active { transform:scale(0.98) !important; }
[data-testid="stDownloadButton"] > button {
    background:#f0fdf4 !important; color:#166534 !important;
    border:1px solid #86efac !important;
}
[data-testid="stDownloadButton"] > button:hover { background:#dcfce7 !important; opacity:1 !important; }
hr { border:none !important; border-top:1px solid #e2e8f0 !important; margin:16px 0 !important; }
.page-title    { display:flex; align-items:center; gap:10px; font-size:20px; font-weight:600; color:#0f172a; margin-bottom:2px; animation:fadeSlideDown 0.4s ease 0.2s both; }
.page-caption  { font-size:12px; color:#64748b; margin-bottom:14px; animation:fadeSlideDown 0.4s ease 0.25s both; }
.section-title { display:flex; align-items:center; gap:7px; font-size:14px; font-weight:600; color:#0f172a; margin-bottom:10px; }
[data-testid="metric-container"] { background:#f8fafc !important; border-radius:10px !important; padding:12px 16px !important; border:0.5px solid #e2e8f0 !important; animation:popIn 0.35s cubic-bezier(0.4,0,0.2,1) both !important; }
[data-testid="metric-container"]:nth-child(1) { animation-delay:0.35s !important; }
[data-testid="metric-container"]:nth-child(2) { animation-delay:0.45s !important; }
[data-testid="stMetricLabel"] { font-size:11px !important; text-transform:uppercase !important; letter-spacing:0.8px !important; color:#94a3b8 !important; font-family:'DM Mono',monospace !important; }
[data-testid="stMetricValue"] { font-size:22px !important; font-weight:600 !important; color:#0f172a !important; }
[data-testid="stMetricDelta"] { font-size:11px !important; }
[data-testid="stDataFrame"] { border-radius:8px !important; overflow:hidden !important; border:0.5px solid #e2e8f0 !important; animation:fadeSlideUp 0.4s ease both; font-family:'DM Mono',monospace !important; font-size:12px !important; }
.insight-box        { border-radius:10px; padding:12px 16px; font-size:13px; margin-bottom:12px; display:flex; align-items:flex-start; gap:10px; line-height:1.6; animation:fadeSlideUp 0.4s ease both; }
.insight-box.green  { background:#dcfce7; color:#166534; }
.insight-box.blue   { background:#eff6ff; color:#1e40af; }
.insight-box.yellow { background:#fef9c3; color:#78350f; }
.insight-box.red    { background:#fee2e2; color:#991b1b; }
.insight-icon       { font-size:16px; margin-top:2px; flex-shrink:0; }
.gauge-block { background:#f8fafc; border:0.5px solid #e2e8f0; border-radius:10px; padding:10px 14px; margin-bottom:8px; animation:fadeSlideUp 0.4s ease both; }
.gauge-label { font-size:11px; font-weight:500; font-family:'DM Mono',monospace; color:#475569; margin-bottom:5px; display:flex; justify-content:space-between; align-items:center; }
.gauge-sub   { font-size:10px; color:#94a3b8; margin-top:4px; }
.gauge-track { height:6px; background:#e2e8f0; border-radius:3px; overflow:hidden; }
.gauge-fill  { height:100%; border-radius:3px; animation:fillBar 0.8s cubic-bezier(0.4,0,0.2,1) both; }
.gauge-fill.good { background:#22c55e; }
.gauge-fill.warn { background:#f59e0b; }
.gauge-fill.bad  { background:#ef4444; }
.idle-state { display:flex; flex-direction:column; align-items:center; justify-content:center; height:80vh; gap:12px; color:#94a3b8; font-size:13px; text-align:center; animation:fadeSlideUp 0.4s ease 0.3s both; }
.idle-icon  { font-size:42px; }
.stSuccess, .stWarning { border-radius:8px !important; font-size:13px !important; animation:fadeSlideDown 0.35s ease both !important; }
.stSpinner { color:#3b82f6 !important; }
</style>
""", unsafe_allow_html=True)
def logo_data_uri(path: Path) -> str | None:
    if not path.is_file():
        return None
    ext = path.suffix.lower().lstrip(".")
    mime = "jpeg" if ext in {"jpg", "jpeg"} else "png" if ext == "png" else "jpeg"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/{mime};base64,{b64}"
# ─── HELPERS ──────────────────────────────────────────────────────────────────
def ks_delta(avg_ks):
    if avg_ks < 0.10:   return "⭐ Excellent", "normal"
    elif avg_ks < 0.20: return "🟢 Good", "normal"
    elif avg_ks < 0.30: return "🟠 Moderate", "off"
    elif avg_ks < 0.40: return "🟡 Acceptable", "off"
    else:               return "🔴 Poor — retrain", "inverse"
def highlight_ks_row(avg_ks):
    if avg_ks < 0.10:   return "0.00 - 0.10"
    elif avg_ks < 0.20: return "0.10 - 0.20"
    elif avg_ks < 0.30: return "0.20 - 0.30"
    elif avg_ks < 0.40: return "0.30 - 0.40"
    else:               return "0.40+"
def style_interp_table(df, active_range):
    def row_style(row):
        if row["Range"] == active_range:
            return ["background-color:#fef9c3; font-weight:600; color:#78350f"] * len(row)
        return [""] * len(row)
    return df.style.apply(row_style, axis=1)
def normalise_score(val) -> float:
    if isinstance(val, dict):
        val = list(val.values())[0]
    if isinstance(val, pd.DataFrame):
        val = val.iloc[0, 0]
    if hasattr(val, "item"):
        val = val.item()
    return float(val)
def clip_numeric_df(df: pd.DataFrame, decimals: int = 4) -> pd.DataFrame:
    out = df.copy()
    num_cols = out.select_dtypes(include="number").columns
    out[num_cols] = out[num_cols].round(decimals)
    return out
def format_memorization_df(results: dict) -> pd.DataFrame:
    exact_pct = float(results.get("exact_memorization", 0.0))
    status = "Safe" if exact_pct == 0.0 else ("Review" if exact_pct < 1.0 else "Leaked")
    return pd.DataFrame([{
        "Check": "Exact duplicate rows",
        "Match %": f"{exact_pct:.2f}%",
        "Status": status,
    }])
def memorization_insight(results: dict):
    exact_pct = float(results.get("exact_memorization", 0.0))
    if exact_pct == 0.0:
        return "green", "🛡️", (
            "<strong>0% of rows memorized or leaked.</strong> No exact matches found between "
            "the real and synthetic datasets. Your synthetic data is safe to share."
        )
    elif exact_pct < 1.0:
        return "yellow", "⚠️", (
            f"<strong>{exact_pct:.2f}% of rows have exact matches</strong> in the real dataset. "
            "Review flagged rows before sharing synthetic data externally."
        )
    else:
        return "red", "🚨", (
            f"<strong>{exact_pct:.2f}% of rows were memorized.</strong> Significant overlap detected. "
            "Consider retraining with stronger privacy constraints."
        )
def correlation_insight(quality_score: float):
    pct = round(float(quality_score) * 100, 1)
    if pct >= 85:
        word, style, icon = "successfully captured", "blue", "🧠"
    elif pct >= 65:
        word, style, icon = "partially captured", "yellow", "⚠️"
    else:
        word, style, icon = "struggled to capture", "red", "🔴"
    return style, icon, (
        f"<strong>Learned {pct}% of feature relationships and patterns.</strong> "
        f"The synthetic model {word} the statistical dependencies "
        "between features in the original dataset."
    )
def render_gauge(label, value, subtitle="", max_val=1.0):
    value = float(value)
    pct = min(value / max_val, 1.0) * 100
    cls = "good" if pct >= 65 else ("warn" if pct >= 40 else "bad")
    sub = f'<div class="gauge-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(f"""
    <div class="gauge-block">
        <div class="gauge-label">
            <span>{label}</span>
            <span style="font-weight:600;color:#0f172a;">{pct:.1f}%</span>
        </div>
        <div class="gauge-track">
            <div class="gauge-fill {cls}" style="width:{pct:.1f}%;"></div>
        </div>
        {sub}
    </div>
    """, unsafe_allow_html=True)
def render_insight(icon, text, style="blue"):
    st.markdown(f"""
    <div class="insight-box {style}">
        <span class="insight-icon">{icon}</span>
        <div>{text}</div>
    </div>
    """, unsafe_allow_html=True)
# ─── LAYOUT ───────────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 1.2], gap="medium")
SCROLL_HEIGHT = 780
with left_col:
    with st.container(height=SCROLL_HEIGHT, border=False):
        st.markdown('<div class="left-pane">', unsafe_allow_html=True)
        _logo_src = logo_data_uri(LOGO_PATH)
        if _logo_src:
            st.markdown(f"""
            <div class="logo-block">
                <img class="logo-img" src="{_logo_src}" alt="SynTab logo" />
                <div>
                    <p class="logo-title">SynTab</p>
                    <p class="logo-sub">Synthetic Tabular Data Engine</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="logo-block">
                <div style="width:46px;height:46px;display:flex;align-items:center;justify-content:center;background:#fee2e2;border-radius:50%;font-size:12px;color:#991b1b;">!</div>
                <div>
                    <p class="logo-title">SynTab</p>
                    <p class="logo-sub">Logo missing ({LOGO_PATH})</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('<div class="tagline">Generate. Evaluate. Trust Synthetic Data.</div>', unsafe_allow_html=True)
        checks = [
            ("✔", "Create realistic synthetic datasets"),
            ("✔", "Evaluate quality using statistical + ML metrics"),
            ("✔", "Compare real vs synthetic data structure"),
            ("✔", "Ensure privacy through memorization checks"),
        ]
        for i, (icon, text) in enumerate(checks):
            st.markdown(f"""
            <div class="check-row" style="animation-delay:{0.25 + i*0.07}s;">
                <span class="ck">{icon}</span> {text}
            </div>
            """, unsafe_allow_html=True)
        st.markdown('<p class="sec-label">Upload dataset</p>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload CSV or Excel file",
            type=["csv", "xlsx"],
            label_visibility="collapsed",
        )
        st.info("Only tabular datasets (CSV/Excel) are supported.")
        if uploaded_file is not None:
            df = load_data(uploaded_file)
            real_df = df.copy()
            st.markdown('<p class="sec-label">Dataset preview</p>', unsafe_allow_html=True)
            st.dataframe(df.head(), use_container_width=True)
            st.caption(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
            k = st.slider("Features to visualize", 1, 10, 3)
            if st.button("▶  Generate Synthetic Data"):
                with st.spinner("Processing… this may take a few minutes."):
                    (X_num, X_cat, sc, label_encoders, num_features,
                     num_imputer, cat_imputer, num_cols, cat_cols,
                     vocab_sizes) = preprocess(df)
                    train_vae(
                        X_num, X_cat, sc, label_encoders, num_features,
                        num_imputer, cat_imputer, num_cols, cat_cols, vocab_sizes
                    )
                    n_samples = len(df)
                    generated_df = generate_full_dataset(n_samples)
                    ks_df, avg_ks, similarity_score = ks_evaluation(real_df, generated_df, num_cols)
                    if avg_ks < 0.4:
                        st.success("✅ VAE output accepted")
                        final_df = generated_df
                        model = "VAE"
                    else:
                        st.warning("⚠️ VAE not sufficient — switching to TVAE")
                        final_df = run_synthesizer(real_df)
                        model = "TVAE"
                    ks_df, avg_ks, similarity_score = ks_evaluation(real_df, final_df, num_cols)
                    mem_results = memorization_check(real_df, final_df)
                    quality_score = normalise_score(correlation_compute(real_df, final_df))
                    final_df.to_csv("data/New/generated_data.csv", index=False)
                st.session_state.update({
                    "df_real": real_df,
                    "df_final": final_df,
                    "model": model,
                    "ks_df": ks_df,
                    "avg_ks": float(avg_ks),
                    "similarity_score": float(similarity_score),
                    "memorization": mem_results,
                    "correlation": quality_score,
                    "k": k,
                    "num_cols": num_cols,
                })
            if "df_final" in st.session_state:
                st.markdown('<p class="sec-label">Export</p>', unsafe_allow_html=True)
                with open("data/New/generated_data.csv", "rb") as f:
                    csv_bytes = f.read()
                st.download_button(
                    label="⬇  Download Synthetic CSV",
                    data=csv_bytes,
                    file_name="synthetic_data.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
        st.markdown('</div>', unsafe_allow_html=True)
with right_col:
    with st.container(height=SCROLL_HEIGHT, border=False):
        st.markdown('<div class="right-pane">', unsafe_allow_html=True)
        if "df_final" not in st.session_state:
            st.markdown("""
            <div class="idle-state">
                <div class="idle-icon">📊</div>
                <div>Upload a dataset and click<br>
                <strong>Generate Synthetic Data</strong> to see metrics here.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            avg_ks = st.session_state["avg_ks"]
            similarity = st.session_state["similarity_score"]
            correlation = st.session_state["correlation"]
            model = st.session_state["model"]
            ks_df = st.session_state["ks_df"]
            memorization = st.session_state["memorization"]
            real_df = st.session_state["df_real"]
            final_df = st.session_state["df_final"]
            num_cols = st.session_state["num_cols"]
            k = st.session_state["k"]
            ks_label, ks_delta_dir = ks_delta(avg_ks)
            st.markdown(f"""
            <div class="page-title">📊 Metrics</div>
            <div class="page-caption">Model used: <strong>{model}</strong></div>
            """, unsafe_allow_html=True)
            st.divider()
            st.markdown('<div class="section-title">📐 KS Test Results</div>', unsafe_allow_html=True)
            m1, m2 = st.columns(2)
            with m1:
                st.metric("Avg KS Statistic", f"{avg_ks:.4f}", delta=ks_label, delta_color=ks_delta_dir)
            with m2:
                st.metric("Similarity Score", f"{similarity * 100:.2f}%", delta="higher is better", delta_color="normal")
            st.dataframe(clip_numeric_df(ks_df), use_container_width=True, hide_index=True)
            st.markdown("##### How to read your Avg KS score")
            ks_interpretation = pd.DataFrame({
                "Range": ["0.00 - 0.10", "0.10 - 0.20", "0.20 - 0.30", "0.30 - 0.40", "0.40+"],
                "Rating": ["⭐ Excellent", "🟢 Good", "🟠 Moderate", "🟡 Acceptable (minimum usable)", "🔴 Poor"],
                "Meaning": [
                    "Near perfect synthetic match",
                    "High quality, minor drift",
                    "Still usable, slight distribution shift",
                    "Bare minimum usable quality",
                    "Not usable, retrain required",
                ],
            })
            active_range = highlight_ks_row(avg_ks)
            st.dataframe(style_interp_table(ks_interpretation, active_range), use_container_width=True, hide_index=True)
            st.divider()
            st.markdown('<div class="section-title">📊 KS Score per Feature</div>', unsafe_allow_html=True)
            ks_fig = plot_ks_summary(ks_df)
            st.pyplot(ks_fig, use_container_width=True)
            st.divider()
            st.markdown('<div class="section-title">📈 Feature Distributions (Top Features)</div>', unsafe_allow_html=True)
            dist_fig = plot_top_k_distributions(real_df, final_df, ks_df, k=k, model_name=model)
            st.pyplot(dist_fig, use_container_width=True)
            st.divider()
            st.markdown('<div class="section-title">🔒 Memorization Check</div>', unsafe_allow_html=True)
            st.dataframe(format_memorization_df(memorization), use_container_width=True, hide_index=True)
            mem_style, mem_icon, mem_text = memorization_insight(memorization)
            render_insight(mem_icon, mem_text, style=mem_style)
            st.divider()
            st.markdown('<div class="section-title">🔗 Correlation Quality</div>', unsafe_allow_html=True)
            corr_style, corr_icon, corr_text = correlation_insight(correlation)
            render_insight(corr_icon, corr_text, style=corr_style)
            render_gauge(
                "Correlation score", correlation,
                subtitle="How well inter-feature relationships are preserved in synthetic data",
            )
            render_gauge(
                "Distribution similarity", similarity,
                subtitle="Based on KS test — how closely each feature distribution matches real data",
            )
            exact_pct = float(memorization.get("exact_memorization", 0.0))
            privacy_score = 1.0 - (exact_pct / 100.0)
            render_gauge(
                "Privacy score", privacy_score,
                subtitle="Percentage of synthetic rows with no exact duplicate in the real dataset",
            )
        st.markdown('</div>', unsafe_allow_html=True)