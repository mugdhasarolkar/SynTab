import tensorflow as tf
from keras import layers,Model
import numpy as np
def build_vae(input_dim):
    h1=min(128,max(8,input_dim//2))
    h2=min(64,max(4,input_dim//4))
    latent_dim=max(2,input_dim//8)
    #Encoder
    encoder_input=tf.keras.Input(shape=(input_dim,),name="encoder_input")
    x=layers.Dense(h1,activation="relu")(encoder_input)
    x=layers.Dense(h2,activation="relu")(x)
    mu=layers.Dense(latent_dim,name="latent_mu")(x)
    log_var=layers.Dense(latent_dim,name="latent_log_var")(x)

    def sampling(args):
        mu,log_var=args
        epsilon=tf.random.normal(shape=tf.shape(mu))
        return mu+tf.exp(0.5*log_var)*epsilon
    z=layers.Lambda(sampling,name="z")([mu,log_var])
    encoder=Model(encoder_input,[mu,log_var],name="encoder")

    #Decoder
    latent_input=layers.Input(shape=(latent_dim,),name="z_sampling")
    x=layers.Dense(h2,activation="relu")(latent_input)
    x=layers.Dense(h1,activation="relu")(x)
    output=layers.Dense(input_dim,activation="sigmoid")(x)
    decoder=Model(latent_input,output,name="decoder")
    vae_output=decoder(encoder(encoder_input)[2])
    vae=Model(encoder_input,vae_output,name="vae")
    #add losses
    kl_loss=-0.5*tf.reduce_mean(tf.reduce_sum(1+log_var-tf.square(mu)-tf.exp(log_var),axis=1))
    reconstruction_loss=tf.reduce_mean(tf.reduce_sum(tf.square(encoder_input-vae_output),axis=1))
    vae.add_loss(reconstruction_loss+kl_loss)
    return vae,encoder,decoder