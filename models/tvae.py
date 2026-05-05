from sdv.single_table import TVAESynthesizer, GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata
import pandas as pd


def run_synthesizer(real_raw_df):

    df = real_raw_df.copy()

    # 🔹 Store schema
    original_dtypes = df.dtypes.to_dict()
    original_columns = df.columns.tolist()

    # 🔹 Identify numeric columns
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # 🔹 Clean numeric columns
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna().reset_index(drop=True)

    # 🔹 Metadata
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)

    n_rows = len(df)

    # 🔹 Model selection
    if n_rows < 50000:
        model = TVAESynthesizer(metadata, epochs=40)
        train_df = df if n_rows < 10000 else df.sample(5000, random_state=42)
    else:
        model = GaussianCopulaSynthesizer(metadata)
        train_df = df

    # 🔹 Train & generate
    model.fit(train_df)
    synth_df = model.sample(n_rows)

    # 🔹 Enforce schema
    synth_df = synth_df.reindex(columns=original_columns)

    for col in num_cols:
        synth_df[col] = pd.to_numeric(synth_df[col], errors='coerce')

    for col, dtype in original_dtypes.items():
        try:
            if "int" in str(dtype):
                synth_df[col] = synth_df[col].round().astype(dtype)
            else:
                synth_df[col] = synth_df[col].astype(dtype)
        except:
            pass

    synth_df = synth_df.dropna().reset_index(drop=True)

    return synth_df