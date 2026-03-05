from preprocessing.preprocess import load_data
from preprocessing.preprocess import preprocess
data_path=r"E:\MyMLproject\syntab\data\raw\loan_data.csv"
df=load_data(data_path)
df_processed,scaler,encoder,num_imputer,cat_imputer=preprocess(df)
X=df_processed.to_numpy()
input_dim=X.shape[1]
print(type(X))
print(X.shape)