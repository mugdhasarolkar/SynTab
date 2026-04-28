import pandas as pd
import joblib
from sklearn.preprocessing import QuantileTransformer, OneHotEncoder
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


# 🔥 NEW: Detect and drop identifier-like columns (general solution)
def drop_identifier_columns(df, threshold=0.9):
    cols_to_drop = []
    
    for col in df.columns:
        unique_ratio = df[col].nunique() / len(df)
        if unique_ratio > threshold:
            cols_to_drop.append(col)

    if cols_to_drop:
        print("Dropping identifier-like columns:", cols_to_drop)

    return df.drop(columns=cols_to_drop)


# 🔥 Reduce high-cardinality categories (less aggressive)
def reduce_categories(df, cat_cols, max_categories=100):
    for col in cat_cols:
        if df[col].nunique() > max_categories:
            top_categories = df[col].value_counts().nlargest(max_categories).index
            df[col] = df[col].where(df[col].isin(top_categories), 'Other')
    return df


# 🔥 Drop extreme high-cardinality columns (fallback safety)
def drop_high_cardinality(df, threshold=1000):
    cols_to_drop = []
    for col in df.select_dtypes(include=['object', 'category']):
        if df[col].nunique() > threshold:
            cols_to_drop.append(col)

    if cols_to_drop:
        print("Dropping high-cardinality columns:", cols_to_drop)

    return df.drop(columns=cols_to_drop)


# 🔥 Safety: limit total categories (prevents explosion)
def limit_total_categories(df, cat_cols, max_total=5000):
    total = sum(df[col].nunique() for col in cat_cols)

    if total > max_total:
        print(f"Too many total categories ({total}), reducing...")
        for col in cat_cols:
            top = df[col].value_counts().nlargest(50).index
            df[col] = df[col].where(df[col].isin(top), 'Other')

    return df


def preprocess(df):
    n = len(df)

    # ✅ Step 1: Sampling
    if n > 45000:
        df = df.sample(n=45000, random_state=42)

    # ✅ Step 2: Drop obvious ID columns
    for col in df.columns:
        if "id" in col.lower():
            df = df.drop(col, axis=1)

    # 🔥 Step 3: Drop identifier-like columns (generalized)
    df = drop_identifier_columns(df)

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

    # 🔥 Step 4: Handle categorical explosion
    df = drop_high_cardinality(df)
    cat_cols = [col for col in cat_cols if col in df.columns]

    df = reduce_categories(df, cat_cols, max_categories=100)
    df = limit_total_categories(df, cat_cols, max_total=5000)

    # 🔥 Step 5: Fill categorical NaNs properly
    df[cat_cols] = df[cat_cols].fillna("Unknown")

    # Constraints for numeric
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

    # Imputation
    num_imputer = SimpleImputer(strategy="mean")
    cat_imputer = SimpleImputer(strategy="most_frequent")

    if df.isnull().values.any():
        df[num_cols] = num_imputer.fit_transform(df[num_cols])
        df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

    # Scaling
    sc =  QuantileTransformer(
    output_distribution='normal',
    n_quantiles=min(1000, len(df)),
    random_state=42)
    df[num_cols] = sc.fit_transform(df[num_cols])

    # Encoding
    encoder = OneHotEncoder(
        sparse_output=False,
        handle_unknown='ignore'
    )

    encoded = encoder.fit_transform(df[cat_cols])

    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(cat_cols),
        index=df.index
    )

    df_processed = pd.concat([df[num_cols], encoded_df], axis=1)

    return df_processed, sc, encoder, num_features, num_imputer, cat_imputer