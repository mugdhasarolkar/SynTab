from preprocessing.preprocess import load_data
from preprocessing.preprocess import preprocess
from models.vae_model import build_vae
from models.train import train_vae
from models.generate import generate_full_dataset
from models.tvae import run_synthesizer
from evaluation.evaluate import ks_evaluation,memorization_check,correlation_compute
from evaluation.plot import plot_ks_summary,plot_top_k_distributions,get_top_k_features
import pandas as pd
data_path=r"C:\Users\Mugdha\Downloads\customer.csv"
df = pd.read_csv(data_path)
real_df=pd.read_csv(data_path)
X_num,X_cat,sc,label_encoders,num_features,num_imputer,cat_imputer,num_cols,cat_cols,vocab_sizes= preprocess(df)

train_vae(X_num,X_cat,sc,label_encoders,num_features,num_imputer,cat_imputer,num_cols,cat_cols,vocab_sizes)
n_samples = len(df)
generated_df = generate_full_dataset(n_samples)
print(generated_df.head())
print(generated_df.shape)
generated_df.to_csv("data/New/generated_data.csv", index=False)
print(" Synthetic data saved successfully")
ks_df, avg_ks, similarity_score=ks_evaluation(real_df,generated_df,num_cols)
print(ks_df)
print(avg_ks)
if avg_ks<0.4:
    plot_ks_summary(ks_df)
    get_top_k_features(ks_df)
    plot_top_k_distributions(real_df,generated_df,ks_df)
    results=memorization_check(real_df,generated_df)
    print(results)
    quality_score=correlation_compute(real_df,generated_df)
    print(quality_score)
    
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
    results=memorization_check(real_df,synth1_df)
    print(results)
    quality_score=correlation_compute(real_df,synth1_df)
    print(quality_score)





