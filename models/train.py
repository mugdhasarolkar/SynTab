from models.vae_model import build_vae
import joblib
def train_vae(x_train):
    input_dim=x_train.shape[1]
    vae,encoder,decoder=build_vae(input_dim)
    vae.compile(optimizer='adam')
    vae.fit(x_train,epochs=30,batch_size=32)
    decoder.save("outputs/saved_models/vae_decoder.h5")
    print("Training complete. decoder saved.")