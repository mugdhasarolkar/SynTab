from preprocessing.preprocess import load_data
from preprocessing.preprocess import preprocess
from models.vae_model import build_vae
from models.train import train_vae
from models.generate import generate_full_dataset
data_path = r"E:\MyMLproject\syntab\data\raw\loan_data.csv"
df = load_data(data_path)
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
