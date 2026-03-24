import tensorflow as tf
from keras import ops
from keras import layers, Model

def build_vae(input_dim):
    h1 = min(128, max(8, input_dim // 2))
    h2 = min(64, max(4, input_dim // 4))
    latent_dim = max(2, input_dim // 8)

    # Encoder
    encoder_input = tf.keras.Input(shape=(input_dim,), name="encoder_input")

    x = layers.Dense(h1, activation="relu")(encoder_input)
    x = layers.Dense(h2, activation="relu")(x)

    mu = layers.Dense(latent_dim, name="latent_mu")(x)
    log_var = layers.Dense(latent_dim, name="latent_log_var")(x)

    def sampling(args):
        mu, log_var = args
        epsilon = tf.random.normal(shape=tf.shape(mu))
        return mu + tf.exp(0.5 * log_var) * epsilon

    z = layers.Lambda(sampling, name="z")([mu, log_var])

    encoder = Model(encoder_input, [mu, log_var, z], name="encoder")

    # Decoder
    latent_input = layers.Input(shape=(latent_dim,), name="z_sampling")

    x = layers.Dense(h2, activation="relu")(latent_input)
    x = layers.Dense(h1, activation="relu")(x)
    output = layers.Dense(input_dim, activation="sigmoid")(x)

    decoder = Model(latent_input, output, name="decoder")

    # VAE Model (Custom Training)
    class VAE(Model):
        def __init__(self, encoder, decoder):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder

        def call(self, inputs):
            mu, log_var, z = self.encoder(inputs)
            return self.decoder(z)

        def train_step(self, data):
            with tf.GradientTape() as tape:
                mu, log_var, z = self.encoder(data)
                reconstruction = self.decoder(z)

                # Reconstruction Loss
                reconstruction_loss = ops.mean(
                    ops.sum(ops.square(data - reconstruction), axis=1)
                )

                # KL Divergence Loss
                kl_loss = -0.5 * ops.mean(
                    ops.sum(
                        1 + log_var - ops.square(mu) - ops.exp(log_var),
                        axis=1
                    )
                )

                total_loss = reconstruction_loss + kl_loss

            grads = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

            return {
                "loss": total_loss,
                "reconstruction_loss": reconstruction_loss,
                "kl_loss": kl_loss,
            }

    vae = VAE(encoder, decoder)

    return vae, encoder, decoder,latent_dim