import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from keras.models import load_model


def generate_full_dataset(n_samples):

    decoder = load_model(
        "outputs/saved_models/vae_decoder.keras",
        compile=False
    )

    latent_dim = joblib.load("outputs/saved_models/latent_dim.pkl")
    scaler = joblib.load("outputs/saved_models/scaler.pkl")
    label_encoders = joblib.load("outputs/saved_models/label_encoders.pkl")

    num_cols = joblib.load("outputs/saved_models/num_cols.pkl")
    cat_cols = joblib.load("outputs/saved_models/cat_cols.pkl")
    columns = joblib.load("outputs/saved_models/columns.pkl")
    constraints = joblib.load("outputs/saved_models/constraints.pkl")

    z = np.random.normal(0, 0.7, size=(n_samples, latent_dim))

    outputs = decoder.predict(z, verbose=0)
    if not isinstance(outputs, list):
        outputs = [outputs]

    if num_cols:
        num_output = outputs[0]
        cat_outputs = outputs[1:]

        num_output = np.clip(num_output, -3, 3)
        num_original = scaler.inverse_transform(num_output)

        for i, col in enumerate(num_cols):
            num_original[:, i] = np.clip(
                num_original[:, i],
                constraints[col]["min"],
                constraints[col]["max"]
            )

        num_df = pd.DataFrame(num_original, columns=num_cols)
    else:
        num_df = pd.DataFrame()
        cat_outputs = outputs

    cat_data = {}

    for i, col in enumerate(cat_cols):

        probs = cat_outputs[i]

        # 🔥 temperature scaling (optional but powerful)
        temp = 0.8
        probs = probs ** (1 / temp)

        # 🔥 normalize probabilities (safety)
        probs = probs / probs.sum(axis=1, keepdims=True)

        # 🔥 sample instead of argmax
        indices = [
            np.random.choice(len(p), p=p)
            for p in probs
        ]

        # convert back to original labels
        le = label_encoders[col]
        values = le.inverse_transform(indices)

        cat_data[col] = values

    cat_df = pd.DataFrame(cat_data)
    # Combine
    df_generated = pd.concat([num_df, cat_df], axis=1)

    df_generated = df_generated[columns]

    df_generated.insert(0, "id", range(1, len(df_generated) + 1))
    # Restore dtypes
    dtypes = joblib.load("outputs/saved_models/dtypes.pkl")

    for col in df_generated.columns:
        if col in dtypes:
            dtype = dtypes[col]

            if pd.api.types.is_integer_dtype(dtype):
                df_generated[col] = pd.to_numeric(df_generated[col], errors='coerce')
                df_generated[col] = df_generated[col].round().astype(int)

            elif pd.api.types.is_float_dtype(dtype):
                df_generated[col] = df_generated[col].astype(float)

            elif pd.api.types.is_object_dtype(dtype):
                df_generated[col] = df_generated[col].astype(str)

            elif pd.api.types.is_bool_dtype(dtype):
                df_generated[col] = df_generated[col].astype(bool)

    return df_generated
