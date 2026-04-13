from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata

def run_ctgan(real_raw_df):
    # 🔹 Create metadata
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(real_raw_df)

    model = CTGANSynthesizer(
        metadata,
        epochs=50
    )
    model.fit(real_raw_df)
    synth1_df = model.sample(len(real_raw_df))

    return model, synth1_df