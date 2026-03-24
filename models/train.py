from models.vae_model import build_vae
import joblib

def train_vae(x_train, scaler, encoder, num_features):
    input_dim = x_train.shape[1]
    vae, encoder_model, decoder, latent_dim = build_vae(input_dim)
    vae.compile(optimizer='adam')
    vae.fit(x_train, epochs=30, batch_size=32)
    decoder.save("outputs/saved_models/vae_decoder.h5")
    joblib.dump(latent_dim, "outputs/saved_models/latent_dim.pkl")
    joblib.dump(scaler, "outputs/saved_models/scaler.pkl")
    joblib.dump(encoder, "outputs/saved_models/ohe.pkl") 
    joblib.dump(num_features, "outputs/saved_models/num_features.pkl")
    print("Training complete. Everything saved.")