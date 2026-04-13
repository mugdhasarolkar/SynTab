import tensorflow as tf
from keras import ops
from keras import layers, Model


def build_vae(input_dim):
    # 🔹 Balanced architecture (depth > width for tabular)
    h1 = min(128, max(32, input_dim))
    h2 = min(64, max(16, input_dim // 2))
    h3 = min(32, max(16, input_dim // 4))
    latent_dim = min(32, max(8, input_dim // 4))

    # ===================== Encoder =====================
    encoder_input = tf.keras.Input(shape=(input_dim,), name="encoder_input")

    x = layers.Dense(h1)(encoder_input)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Dense(h2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Dense(h3)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    mu = layers.Dense(latent_dim, name="latent_mu")(x)
    log_var = layers.Dense(latent_dim, name="latent_log_var")(x)

    def sampling(args):
        mu, log_var = args
        epsilon = tf.random.normal(shape=tf.shape(mu))
        return mu + tf.exp(0.5 * log_var) * epsilon

    z = layers.Lambda(sampling, name="z")([mu, log_var])

    encoder = Model(encoder_input, [mu, log_var, z], name="encoder")

    # ===================== Decoder =====================
    latent_input = layers.Input(shape=(latent_dim,), name="z_sampling")

    x = layers.Dense(h3)(latent_input)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Dense(h2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Dense(h1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # 🔹 Keep linear output since you're using StandardScaler
    output = layers.Dense(input_dim, activation="linear")(x)

    decoder = Model(latent_input, output, name="decoder")

    # ===================== VAE =====================
    class VAE(Model):
        def __init__(self, encoder, decoder):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder
            self.kl_weight = 0.0

        def call(self, inputs):
            mu, log_var, z = self.encoder(inputs)
            return self.decoder(z)

        def train_step(self, data):
            # 🔹 KL annealing (more stable, capped lower for tabular)
            self.kl_weight = min(0.1, self.kl_weight + 0.0005)

            # 🔹 Add slight noise (denoising VAE effect)
            noisy_data = data + 0.01 * tf.random.normal(tf.shape(data))

            with tf.GradientTape() as tape:
                mu, log_var, z = self.encoder(noisy_data)
                reconstruction = self.decoder(z)

                # 🔹 Reconstruction Loss (MSE)
                reconstruction_loss = ops.mean(
                    ops.sum(ops.square(data - reconstruction), axis=1)
                )

                # 🔹 KL Divergence
                kl_loss = -0.5 * ops.mean(
                    ops.sum(
                        1 + log_var - ops.square(mu) - ops.exp(log_var),
                        axis=1
                    )
                )

                total_loss = reconstruction_loss + self.kl_weight * kl_loss

            grads = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

            return {
                "loss": total_loss,
                "reconstruction_loss": reconstruction_loss,
                "kl_loss": kl_loss,
                "kl_weight": self.kl_weight,
            }

    vae = VAE(encoder, decoder)

    return vae, encoder, decoder, latent_dim