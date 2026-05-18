import tensorflow as tf
from keras import layers, Model
import re


def build_vae(num_cols, cat_cols, vocab_sizes):

    if not num_cols and not cat_cols:
        raise ValueError("At least one numeric or categorical column is required.")

    safe_col_map = {
        col: re.sub(r"[^A-Za-z0-9_]", "_", col)
        for col in cat_cols
    }

    embedding_dim_total = 0

    for col in cat_cols:
        vocab_size = vocab_sizes[col]
        embed_dim = min(50, vocab_size // 2 + 1)
        embedding_dim_total += embed_dim

    feature_dim = len(num_cols) + embedding_dim_total

    h1 = min(128, max(32, feature_dim))
    h2 = min(64, max(16, feature_dim // 2))
    h3 = min(32, max(16, feature_dim // 4))
    latent_dim = min(16, max(8, feature_dim // 4))

    # =========================
    # Inputs
    # =========================
    num_input = None
    if num_cols:
        num_input = tf.keras.Input(shape=(len(num_cols),), name="num_input")

    cat_inputs = []
    cat_embeddings = []

    for col in cat_cols:
        vocab_size = vocab_sizes[col]
        embed_dim = min(50, vocab_size // 2 + 1)

        safe_col = safe_col_map[col]

        inp = tf.keras.Input(shape=(1,), name=safe_col)

        emb = layers.Embedding(
            input_dim=vocab_size + 1,
            output_dim=embed_dim,
            name=f"{safe_col}_emb"
        )(inp)

        emb = layers.Flatten()(emb)

        cat_inputs.append(inp)
        cat_embeddings.append(emb)

    # =========================
    # Fusion
    # =========================
    if num_cols:
        num_repr = layers.Dense(h2, activation="relu")(num_input)
        num_repr = layers.LayerNormalization()(num_repr)
    else:
        num_repr = None

    if cat_cols:
        if len(cat_embeddings) == 1:
            cat_repr = cat_embeddings[0]
        else:
            cat_repr = layers.Concatenate()(cat_embeddings)
        cat_repr = layers.Dense(h2, activation="relu")(cat_repr)
        cat_repr = layers.LayerNormalization()(cat_repr)
    else:
        cat_repr = None

    if num_repr is not None and cat_repr is not None:
        encoder_input = layers.Concatenate()([num_repr, cat_repr])
    elif num_repr is not None:
        encoder_input = num_repr
    else:
        encoder_input = cat_repr

    # =========================
    # Encoder
    # =========================
    x = layers.Dense(h1)(encoder_input)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Dense(h2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Dense(h3)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.LayerNormalization()(x)

    mu = layers.Dense(latent_dim, name="latent_mu")(x)
    mu = layers.Lambda(lambda t: tf.clip_by_value(t, -3.0, 3.0))(mu)

    log_var = layers.Dense(latent_dim, activation="tanh")(x)
    log_var = log_var * 2.0

    def sampling(args):
        mu, log_var = args
        epsilon = tf.random.normal(shape=tf.shape(mu))
        return mu + tf.exp(0.5 * log_var) * epsilon

    z = layers.Lambda(sampling, name="z")([mu, log_var])

    z = layers.LayerNormalization()(z)
    z = layers.GaussianNoise(0.01)(z)

    encoder_inputs = ([num_input] if num_cols else []) + cat_inputs
    encoder = Model(encoder_inputs, [mu, log_var, z], name="encoder")

    # =========================
    # Decoder (FiLM version you had working)
    # =========================
    latent_input = layers.Input(shape=(latent_dim,), name="z_sampling")

    z = latent_input

    # project z per layer size
    z_h3 = layers.Dense(h3)(z)
    z_h2 = layers.Dense(h2)(z)
    z_h1 = layers.Dense(h1)(z)

    # =========================
    # Block 1
    # =========================
    gamma = layers.Dense(h3)(z_h3)
    beta = layers.Dense(h3)(z_h3)

    h = layers.Dense(h3)(z)
    x = layers.LayerNormalization()(h)
    x = layers.Multiply()([x, gamma + 1.0])
    x = layers.Add()([x, beta])
    x = layers.ReLU()(x)

    skip = x
    x = layers.Dense(h3, activation="relu")(x)
    x = layers.Add()([x, skip])

    # =========================
    # Block 2
    # =========================
    gamma = layers.Dense(h2)(z_h2)
    beta = layers.Dense(h2)(z_h2)

    h = layers.Dense(h2)(x)
    x = layers.LayerNormalization()(h)
    x = layers.Multiply()([x, gamma + 1.0])
    x = layers.Add()([x, beta])
    x = layers.ReLU()(x)

    # =========================
    # Block 3
    # =========================
    gamma = layers.Dense(h1)(z_h1)
    beta = layers.Dense(h1)(z_h1)

    h = layers.Dense(h1)(x)
    x = layers.LayerNormalization()(h)
    x = layers.Multiply()([x, gamma + 1.0])
    x = layers.Add()([x, beta])
    x = layers.ReLU()(x)

    # =========================
    # Outputs (ORIGINAL SAFE)
    # =========================
    decoder_outputs = []
    if num_cols:
        decoder_outputs.append(
            layers.Dense(len(num_cols), name="num_output")(x)
        )

    for col in cat_cols:
        vocab_size = vocab_sizes[col]
        safe_col = safe_col_map[col]

        out = layers.Dense(
            vocab_size,
            activation="softmax",
            name=f"{safe_col}_out"
        )(x)

        decoder_outputs.append(out)

    decoder = Model(latent_input, decoder_outputs, name="decoder")

    # =========================
    # VAE Model (UNCHANGED)
    # =========================
    class VAE(Model):

        def __init__(self, encoder, decoder):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder
            self.kl_weight = 0.0
            self.step = 0

        def call(self, inputs):
            mu, log_var, z = self.encoder(inputs)
            return self.decoder(z)

        def train_step(self, data):

            self.step += 1
            inputs = data[0]

            with tf.GradientTape() as tape:

                mu, log_var, z = self.encoder(inputs)
                reconstruction = self.decoder(z)
                if not isinstance(reconstruction, (list, tuple)):
                    reconstruction = [reconstruction]

                num_loss = tf.constant(0.0, dtype=tf.float32)
                cat_loss = tf.constant(0.0, dtype=tf.float32)
                recon_i = 0
                inp_i = 0

                if num_cols:
                    num_pred = reconstruction[recon_i]
                    recon_i += 1
                    num_true = tf.cast(inputs[inp_i], tf.float32)
                    inp_i += 1
                    num_loss = tf.reduce_mean(
                        tf.reduce_sum(tf.square(num_true - num_pred), axis=1)
                    )

                if cat_cols:
                    cat_preds = reconstruction[recon_i:]
                    cat_true = [
                        tf.cast(tf.reshape(c, (-1,)), tf.int32)
                        for c in inputs[inp_i:]
                    ]
                    for i in range(len(cat_cols)):
                        weight = 1.0 / tf.math.log(
                            tf.cast(vocab_sizes[cat_cols[i]] + 1, tf.float32)
                        )

                        cat_loss += weight * tf.reduce_mean(
                            tf.keras.losses.sparse_categorical_crossentropy(
                                cat_true[i],
                                cat_preds[i]
                            )
                        )

                reconstruction_loss = num_loss + cat_loss

                if self.step < 2000:
                    self.kl_weight = 0.0
                else:
                    self.kl_weight = tf.minimum(
                        0.05,
                        tf.cast(self.step - 2000, tf.float32) / 15000.0
                    )

                kl_per_sample = -0.5 * tf.reduce_sum(
                    1 + log_var - tf.square(mu) - tf.exp(log_var),
                    axis=1
                )

                kl_loss = tf.reduce_mean(tf.clip_by_value(kl_per_sample, 0.0, 20.0))
                kl_loss = kl_loss / tf.cast(latent_dim, tf.float32)

                total_loss = reconstruction_loss + self.kl_weight * kl_loss

            grads = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

            return {
                "loss": total_loss,
                "reconstruction_loss": reconstruction_loss,
                "num_loss": num_loss,
                "cat_loss": cat_loss,
                "kl_loss": kl_loss,
                "kl_weight": self.kl_weight,
            }

    vae = VAE(encoder, decoder)

    return vae, encoder, decoder, latent_dim
