from models.vae_model import build_vae
import joblib


def train_vae(X_num, X_cat, sc, label_encoders, num_features, num_imputer, cat_imputer, num_cols, cat_cols, vocab_sizes):

    vae, encoder_model, decoder, latent_dim = build_vae(
        num_cols,
        cat_cols,
        vocab_sizes
    )

    vae.compile(optimizer='adam')

    x_train = []
    if num_cols:
        x_train.append(X_num.values.astype("float32"))

    for col in cat_cols:
        x_train.append(
            X_cat[col]
            .values
            .reshape(-1, 1)
            .astype("int32")
        )

    # Train
    vae.fit(
        x_train,
        epochs=30,
        batch_size=32
    )

    # Save models + preprocessors
    decoder.save(
        "outputs/saved_models/vae_decoder.keras"
    )

    joblib.dump(
        latent_dim,
        "outputs/saved_models/latent_dim.pkl"
    )

    joblib.dump(
        sc,
        "outputs/saved_models/scaler.pkl"
    )

    joblib.dump(
        label_encoders,
        "outputs/saved_models/label_encoders.pkl"
    )

    joblib.dump(
        num_features,
        "outputs/saved_models/num_features.pkl"
    )

    joblib.dump(
        num_imputer,
        "outputs/saved_models/num_imputer.pkl"
    )

    joblib.dump(
        cat_imputer,
        "outputs/saved_models/cat_imputer.pkl"
    )

    joblib.dump(
        vocab_sizes,
        "outputs/saved_models/vocab_sizes.pkl"
    )

    print(
        "Training complete. Everything saved."
    )
