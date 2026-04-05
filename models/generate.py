import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from keras.models import load_model


def sampling(args):
    mu, log_var = args
    epsilon = tf.random.normal(shape=tf.shape(mu))
    return mu + tf.exp(0.5 * log_var) * epsilon


def generate_full_dataset(n_samples):

    decoder = load_model(
        "outputs/saved_models/vae_decoder.h5",
        custom_objects={"sampling": sampling},
        compile=False
    )

    latent_dim = joblib.load("outputs/saved_models/latent_dim.pkl")
    num_features = joblib.load("outputs/saved_models/num_features.pkl")
    scaler = joblib.load("outputs/saved_models/scaler.pkl")
    ohe = joblib.load("outputs/saved_models/ohe.pkl")
    columns = joblib.load("outputs/saved_models/columns.pkl")
    num_cols = joblib.load("outputs/saved_models/num_cols.pkl")
    cat_cols = joblib.load("outputs/saved_models/cat_cols.pkl")
    constraints = joblib.load("outputs/saved_models/constraints.pkl")
    z = np.random.normal(loc=0, scale=1.0, size=(n_samples, latent_dim))
    generated = decoder.predict(z, verbose=0)
    num_part = generated[:, :num_features]
    cat_part = generated[:, num_features:]
    cat_part = np.clip(cat_part, 0, 1)
    num_original = scaler.inverse_transform(num_part)
    for i, col in enumerate(num_cols):
        num_original[:, i] = np.clip(
            num_original[:, i],
            constraints[col]["min"],
            constraints[col]["max"]
        )
    cat_original = ohe.inverse_transform(cat_part)
    num_df = pd.DataFrame(num_original, columns=num_cols)
    cat_df = pd.DataFrame(cat_original, columns=cat_cols)
    final_data = pd.concat([num_df, cat_df], axis=1)
    df_generated = pd.DataFrame(final_data, columns=columns)
    df_generated.insert(0, "id", range(1, len(df_generated) + 1))
    dtypes = joblib.load("outputs/saved_models/dtypes.pkl")

    for col in df_generated.columns:
        if col in dtypes:
            dtype = dtypes[col]

            if pd.api.types.is_integer_dtype(dtype):
                df_generated[col] = df_generated[col].round().astype(int)

            elif pd.api.types.is_float_dtype(dtype):
                df_generated[col] = df_generated[col].astype(float)

            elif pd.api.types.is_object_dtype(dtype):
                df_generated[col] = df_generated[col].astype(str)

            elif pd.api.types.is_bool_dtype(dtype):
                df_generated[col] = df_generated[col].astype(bool)

    return df_generated