from preprocessing.preprocess import load_data
from preprocessing.preprocess import preprocess
from models.vae_model import build_vae
from models.train import train_vae
from models.generate import generate_full_dataset
from models.vtae import run_synthesizer
from evaluation.evaluate import ks_evaluation
from evaluation.plot import plot_ks_summary,plot_top_k_distributions,get_top_k_features
import joblib
data_path=r"C:\Users\Mugdha\Downloads\Spotify_Music.csv"
df = load_data(data_path)
real_df=load_data(data_path)
df_processed, scaler, encoder, num_features, num_imputer, cat_imputer = preprocess(df)
X = df_processed.to_numpy()
print(type(X))
print(X.shape)
train_vae(X, scaler, encoder, num_features)
n_samples = X.shape[0]
generated_df = generate_full_dataset(n_samples)
print(generated_df.head())
print(generated_df.shape)
generated_df.to_csv("data/New/generated_data.csv", index=False)
print(" Synthetic data saved successfully")
num_cols=joblib.load(r"E:\MyMLproject\syntab\outputs\saved_models\num_cols.pkl")
ks_df, avg_ks, similarity_score=ks_evaluation(real_df,generated_df,num_cols)
print(ks_df)
print(avg_ks)
if avg_ks<0.4:
    plot_ks_summary(ks_df)
    get_top_k_features(ks_df)
    plot_top_k_distributions(real_df,generated_df,ks_df)
else:
    print("VAE not good switching to TVAE")
    synth1_df=run_synthesizer(real_df)
    synth1_df.to_csv("data/New/generated_data.csv", index=False)
    ks_df, avg_ks, similarity_score=ks_evaluation(real_df,synth1_df,num_cols)
    print(ks_df)
    print(avg_ks)
    plot_ks_summary(ks_df)
    get_top_k_features(ks_df)
    plot_top_k_distributions(real_df,synth1_df,ks_df)




