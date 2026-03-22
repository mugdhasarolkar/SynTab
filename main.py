from preprocessing.preprocess import load_data
from preprocessing.preprocess import preprocess
from models.vae_model import build_vae
from models.train import train_vae
data_path=r"E:\MyMLproject\syntab\data\raw\loan_data.csv"
df=load_data(data_path)
df_processed,scaler,encoder,num_imputer,cat_imputer=preprocess(df)
X=df_processed.to_numpy()
input_dim=X.shape[1]
print(type(X))
print(X.shape)
build_vae(input_dim)
train_vae(X)
