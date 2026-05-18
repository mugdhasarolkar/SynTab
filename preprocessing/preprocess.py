import os

import pandas as pd
import joblib
from sklearn.preprocessing import QuantileTransformer, LabelEncoder
from sklearn.impute import SimpleImputer


def load_data(data_input):

    if data_input is None:
        return None

    file_name = data_input.name.lower()

    if file_name.endswith('.csv'):
        df = pd.read_csv(data_input)

    elif file_name.endswith('.xlsx') or file_name.endswith('.xls'):
        df = pd.read_excel(data_input)

    else:
        raise ValueError("Unsupported file format. Use CSV or Excel.")

    return df


def preprocess(df):

    for col in df.columns:
        if "id" in col.lower():
            df = df.drop(col, axis=1)

    if df.shape[1] == 0:
        raise ValueError("No columns left after removing ID columns.")

    os.makedirs("outputs/saved_models", exist_ok=True)

    # Save metadata
    joblib.dump(df.columns.tolist(), "outputs/saved_models/columns.pkl")
    joblib.dump(df.dtypes.to_dict(), "outputs/saved_models/dtypes.pkl")

    # Split columns
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    # Convert low-unique numeric to categorical
    for col in num_cols.copy():
        if df[col].nunique() <= 10:
            cat_cols.append(col)
            num_cols.remove(col)

    constraints = {}

    for col in num_cols:
        constraints[col] = {
            "min": df[col].min(),
            "max": df[col].max()
        }

    joblib.dump(constraints, "outputs/saved_models/constraints.pkl")
    joblib.dump(num_cols, "outputs/saved_models/num_cols.pkl")
    joblib.dump(cat_cols, "outputs/saved_models/cat_cols.pkl")

    num_features = len(num_cols)

    if not num_cols and not cat_cols:
        raise ValueError("No usable numeric or categorical features found.")

    # Imputation
    num_imputer = SimpleImputer(strategy="mean")
    cat_imputer = SimpleImputer(strategy="most_frequent")

    if df.isnull().values.any():
        if num_cols:
            df[num_cols] = num_imputer.fit_transform(df[num_cols])
        if cat_cols:
            df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

    # Scaling
    sc = None
    if num_cols:
        sc = QuantileTransformer(
            output_distribution='normal',
            n_quantiles=min(1000, len(df)),
            random_state=42
        )
        df[num_cols] = sc.fit_transform(df[num_cols])

    # Encoding
    label_encoders = {}
    vocab_sizes = {}

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

        label_encoders[col] = le
        vocab_sizes[col] = df[col].nunique()

    X_num = df[num_cols] if num_cols else pd.DataFrame(index=df.index)
    X_cat = df[cat_cols] if cat_cols else pd.DataFrame(index=df.index)

    return (
        X_num,
        X_cat,
        sc,
        label_encoders,
        num_features,
        num_imputer,
        cat_imputer,
        num_cols,
        cat_cols,
        vocab_sizes
    )
