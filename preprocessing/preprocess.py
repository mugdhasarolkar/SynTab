import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
def load_data(data_path):
    df=pd.read_csv(data_path)
    return df
def preprocess(df):
    for col in df.columns:
        if "id" in col.lower():
            df=df.drop(col,axis=1)
            continue
    joblib.dump(df.columns.tolist(),"outputs/saved_models/columns.pkl")
    num_cols=df.select_dtypes(include=['int64','float64']).columns
    cat_cols=df.select_dtypes(include=['object','category','bool']).columns
    joblib.dump(num_cols,"outputs/saved_models/num_cols.pkl")
    joblib.dump(cat_cols,"outputs/saved_models/cat_cols.pkl")
    num_features=len(num_cols)
    num_imputer=SimpleImputer(strategy="mean")
    cat_imputer=SimpleImputer(strategy="most_frequent")
    if df.isnull().values.any():
        df[num_cols]=num_imputer.fit_transform(df[num_cols])
        df[cat_cols]=cat_imputer.fit_transform(df[cat_cols])
    sc=MinMaxScaler()
    df[num_cols]=sc.fit_transform(df[num_cols])

    encoder=OneHotEncoder(sparse_output=False,handle_unknown='ignore')
    encoded=encoder.fit_transform(df[cat_cols])
    encoded_df=pd.DataFrame(encoded,columns=encoder.get_feature_names_out(cat_cols),index=df.index)
    df_processed=pd.concat([df[num_cols],encoded_df],axis=1)
    return df_processed,sc,encoder,num_features,num_imputer,cat_imputer
