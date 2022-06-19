import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from sklearn.manifold import TSNE
from tensorflow.keras.layers import concatenate as concat
      
def normalize(x, mn, mx):
    return (x-mn)/(mx-mn)

def denormalize(x, mn, mx):
    return x * (mx-mn) + mn
    
def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)
        
class Normalize(layers.Layer):
    def __init__(self, mn=0, mx=1):
        super(Normalize, self).__init__()
        self.mn = self.add_weight(initializer=tf.keras.initializers.Constant(value=mn), name="norm_mn", trainable=False)
        self.mx = self.add_weight(initializer=tf.keras.initializers.Constant(value=mx), name="norm_mx", trainable=False)
    
    def call(self, inputs):
        num = tf.subtract(inputs, self.mn)
        denum = tf.subtract(self.mx, self.mn)
        return tf.divide(num, denum)
    
class Denormalize(layers.Layer):
    def __init__(self, mn=0, mx=1):
        super(Denormalize, self).__init__()
        self.mn = self.add_weight(initializer=tf.keras.initializers.Constant(value=mn), name="denorm_mn", trainable=False)
        self.mx = self.add_weight(initializer=tf.keras.initializers.Constant(value=mx), name="denorm_mx", trainable=False)
    
    def call(self, inputs):
        el1 = tf.subtract(self.mx, self.mn)
        el2 = tf.multiply(inputs, el1)
        return tf.add(el2, self.mn)

class Sampling(layers.Layer):
    """
    Uses (z_mean, z_log_var) to sample z, the vector encoding a digit.
    """

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim), seed=0)
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
  

        
    
#%% VAE

class VAE(keras.Model):
    
    def __init__(self, encoder, decoder, seg_size, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.seg_size = seg_size
        
    def train_step(self, data):
        with tf.GradientTape() as tape:
            x_normal, z_mean, z_log_var, z_conditional = self.encoder(data)
            reconstruction_normal, reconstruction = self.decoder(z_conditional)
            reconstruction_loss = tf.reduce_sum(
                keras.losses.binary_crossentropy(x_normal, reconstruction_normal)
            )
            # reconstruction_loss *= 1 * self.seg_size
            kl_loss = -1 - z_log_var + tf.square(z_mean) + tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= 0.5
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }
    
class CVAE_model():
    def __init__(self, latent_dim, seg_size, channel_nb, class_nb, min_train, max_train,
                 arch='1', show_summary=False):
        self.seg_size = seg_size
        self.channel_nb = channel_nb
        self.class_nb = class_nb
        self.arch = arch
        self.latent_dim = latent_dim
        self.min_train = min_train
        self.max_train = max_train
        if arch=='FCN':
            #Encoder            
            X_layer = keras.Input(shape=(self.seg_size*self.channel_nb,))
            normalized_X = tf.reshape(Normalize(min_train, max_train)(X_layer), ((-1, self.seg_size, self.channel_nb)))
            label_layer = keras.Input(shape=(self.class_nb,))
            encoder_inputs = concat([normalized_X, label_layer])
            x = layers.Dense(500, activation="relu", name='enc_dense1')(encoder_inputs)
            print(X_layer.shape, normalized_X.shape)
            x = layers.Dense(250, activation="relu", name='enc_dense2')(x)
            z_mean = layers.Dense(latent_dim, name="z_mean")(x)
            z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
            z = Sampling()([z_mean, z_log_var])
            z_conditional = concat([z, label_layer])
            print(X_layer.shape, normalized_X.shape)
            self.encoder = keras.Model([X_layer, label_layer], [normalized_X, z_mean, z_log_var, z_conditional], name="encoder")
            #Decoder
            latent_inputs = keras.Input(shape=(latent_dim+self.class_nb,))
            x = layers.Dense(250, activation="relu", name='dec_dense1')(latent_inputs)
            x = layers.Dense(500, activation="relu", name='dec_dense2')(x)
            decoder_outputs = layers.Dense(self.seg_size*self.channel_nb, activation="relu", name='dec_output')(x)
            denormalized_output = Denormalize(min_train, max_train)(decoder_outputs)
            self.decoder = keras.Model(latent_inputs, [decoder_outputs, denormalized_output], name="decoder")
            self.vae = VAE(self.encoder, self.decoder, self.seg_size)
            if show_summary:
                self.encoder.summary()
                self.decoder.summary()
                
        elif arch=='LSTM':
            #Encoder            
            X_layer = keras.Input(shape=(self.seg_size, self.channel_nb))
            label_layer = keras.Input(shape=(self.class_nb,))
            x = layers.LSTM(128, activation="relu", return_sequences=True, name='enc_lstm1')(X_layer)
            x = layers.LSTM(64,  activation="relu", return_sequences=False, name='enc_lstm2')(x)
            # x = layers.RepeatVector(self.seg_size, name='enc_rv2')(x)
            concat_layer = concat([x, label_layer])
            x = layers.Dense(100, activation="relu", name='enc_dense1')(concat_layer)
            z_mean = layers.Dense(latent_dim, name="z_mean")(x)
            z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
            z = Sampling()([z_mean, z_log_var])
            z_conditional = concat([z, label_layer])
            self.encoder = keras.Model([X_layer, label_layer], [z_mean, z_log_var, z_conditional], name="encoder")
            #Decoder
            latent_inputs = keras.Input(shape=(latent_dim+self.class_nb,))
            x = layers.Dense(100, activation="relu", name='dec_dense1')(latent_inputs)
            x = layers.RepeatVector(self.seg_size, name='dec_rv2')(x)
            x =  layers.LSTM(64,  activation="relu", return_sequences=True, name='dec_lstm1')(x)
            x =  layers.LSTM(128,  activation="relu", return_sequences=True, name='dec_lstm2')(x)
            decoder_outputs = layers.TimeDistributed(layers.Dense(self.channel_nb), name='dec_output')(x)
            self.decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
            self.vae = VAE(self.encoder, self.decoder, self.seg_size)
            if show_summary:
                self.encoder.summary()
                self.decoder.summary()
                
                
        elif arch=='CONV':
            #Encoder
            X_layer = keras.Input(shape=(self.seg_size, self.channel_nb))
            normalized_X = Normalize(min_train, max_train)(X_layer)
            # normalized_X = X_layer
            label_layer = keras.Input(shape=(self.class_nb,))
            x = layers.Conv1D(filters=16, kernel_size=2, activation="relu", strides=1, padding="same", name="enc_conv1")(normalized_X)
            x = layers.Conv1D(filters=32, kernel_size=3, activation="relu", strides=1, padding="same", name="enc_conv2")(x)
            l_size = 64
            # assert l_size <= seg_size, "l_size value too large for this data"
            x = layers.Conv1D(filters=l_size, kernel_size=2, activation="relu", strides=1, padding="same", name="enc_conv3")(x)
            x = layers.Flatten(name="enc_flat")(x)
            concat_layer = concat([x, label_layer])
            x = layers.Dense(l_size, activation="relu", name="enc_dense")(concat_layer)
            z_mean = layers.Dense(latent_dim, name="z_mean")(x)
            z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
            z = Sampling()([z_mean, z_log_var])
            z_conditional = concat([z, label_layer])
            self.encoder = keras.Model([X_layer, label_layer], [normalized_X, z_mean, z_log_var, z_conditional], name="encoder")
            #Decoder
            flat_size = self.encoder.get_layer("enc_flat").output_shape[1]
            k_size = np.int(flat_size/l_size)
            latent_inputs = keras.Input(shape=(latent_dim+self.class_nb,))
            x = layers.Dense(k_size * l_size, activation="relu", name="dec_dense")(latent_inputs)
            x = layers.Reshape((k_size, l_size))(x)
            x = layers.Conv1DTranspose(filters=l_size, kernel_size=2, activation="relu", strides=1, padding="same", name="dec_conv1")(x)
            x = layers.Conv1DTranspose(filters=32, kernel_size=3, activation="relu", strides=1, padding="same", name="dec_conv2")(x)
            x = layers.Conv1DTranspose(filters=16, kernel_size=2, activation="relu", strides=1, padding="same", name="dec_conv3")(x)
            decoder_outputs = layers.Conv1DTranspose(filters=self.channel_nb, kernel_size=3, activation="sigmoid", padding="same", name="dec_output")(x)
            denormalized_output = Denormalize(min_train, max_train)(decoder_outputs)
            # denormalized_output =decoder_outputs
            self.decoder = keras.Model(latent_inputs, [decoder_outputs, denormalized_output], name="decoder")
            self.vae = VAE(self.encoder, self.decoder, self.seg_size)
            if show_summary:
                self.encoder.summary()
                self.decoder.summary()
        else:
            raise ValueError("Architecture unknown")
        
    def train(self, data, labels, checkpoint_path, epochs=300, batch_size=10,
              verbose=0, new_train=False):
        if self.arch=='FCN_UV': data = data.reshape((-1, self.seg_size*self.channel_nb))
        if not new_train:
            self.vae.load_weights(checkpoint_path)
        else:
            self.vae.compile(optimizer=keras.optimizers.Adam(1e-4))
            self.vae.fit(data, labels, epochs=epochs, batch_size=batch_size, verbose=verbose)
            self.vae.save_weights(checkpoint_path)
    
    def predict(self, data, labels):
        if self.arch=='FCN': data = data.reshape((-1, self.seg_size*self.channel_nb))
        _,_,_,encoded_data = self.encoder([data, labels])
        _,data_hat = self.decoder(encoded_data)
        return data_hat.numpy().reshape((-1, self.seg_size, self.channel_nb))
    
    def likelihood(self, data, labels, mc_range=10):
        lls = []
        for x, y in zip(data, labels):
            x = np.expand_dims(x, 0)
            y = np.expand_dims(y, 0)
            x_normal, z_mean, z_log_var, z_conditional = self.encoder([x, y])
            rl = 0
            for mc_i in range(mc_range):
                z = Sampling()([z_mean, z_log_var])
                z_conditional = concat([z, y])
                reconstruction_normal, _ = self.decoder(z_conditional)
                reconstruction_loss = tf.reduce_sum(
                    keras.losses.binary_crossentropy(x_normal, reconstruction_normal)
                )
                rl += -reconstruction_loss
            lls.append(rl/mc_range)
        return np.array(lls)
    
    
    
    def likelihood_max(self, x, y, mc_range=50):
        x_normal, z_mean, z_log_var, z_conditional = self.encoder([x, y])
        max_ll_x = -np.inf
        for mc_i in range(mc_range):
            z = Sampling()([z_mean, z_log_var])
            z_conditional = concat([z, y])
            reconstruction_normal, _ = self.decoder(z_conditional)
            cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=reconstruction_normal, labels=tf.Variable(x, dtype=tf.float32))
            logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2])
            logpz = log_normal_pdf(z, 0., 1.)
            logqz_x = log_normal_pdf(z, z_mean, z_log_var)
            ll_x = logpx_z + logpz - logqz_x
            if ll_x >= max_ll_x:
                max_ll_x = ll_x
        return max_ll_x

    def factor_p_x(self, x, y, factor, mc_range=10):
        '''
        return the equivalent of p(x)/factor
        '''
        x_normal, z_mean, z_log_var, z_conditional = self.encoder([x, y])
        sum_loss = 0
        for mc_i in range(mc_range):
            z = Sampling()([z_mean, z_log_var])
            z_conditional = concat([z, y])
            reconstruction_normal, _ = self.decoder(z_conditional)
            cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=reconstruction_normal, labels=tf.Variable(x, dtype=tf.float32))
            logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2])
            logpz = log_normal_pdf(z, 0., 1.)
            logqz_x = log_normal_pdf(z, z_mean, z_log_var)
            logp_x = logpx_z + logpz - logqz_x
            ep_x = np.exp(logp_x-factor)
            sum_loss += ep_x
        return sum_loss/mc_range
        
    def performance(self, X, Y, order=2):
        metric = lambda x,y: (1/len(x))*(np.sum((x-y)**2))
        print("Performance according to average MSE reconstruction metric: ", end='')
        l2_sum = 0
        for i in range(X.shape[0]):
            x = X[i:i+1]
            y = Y[i:i+1]
            x_hat = self.predict(x,y)
            l2_sum += metric(x[0], x_hat[0])
        perf = l2_sum/X.shape[0]
        print("{:.3f}".format(perf))
        return perf
      
    def normalized_performance(self, X, y, order=2):
        if self.arch=='FCN': X = X.reshape((-1, self.seg_size*self.channel_nb))
        X_norm,_,_,encoded_data = self.encoder([X, y])
        Xhat_norm,_ = self.decoder(encoded_data)        
        Xhat_norm.numpy().reshape((-1, self.seg_size, self.channel_nb)) 
        metric = lambda x,y: (1/len(x))*(np.sum((x-y)**2))
        print("Performance according to average MSE reconstruction metric: ", end='')
        l2_sum = 0
        for i in range(Xhat_norm.shape[0]):
            x = X_norm[i:i+1]
            x_hat = Xhat_norm[i:i+1]
            l2_sum += metric(x[0], x_hat[0])
        perf = l2_sum/X.shape[0]
        print("{:.3f}".format(perf))
        return perf
        
    def plot_label_clusters(self, data, name=""):
    # Display how the latent space clusters different classes
        labels = np.argmax(data[1], axis=1)
        _,z_mean, _, _ = self.encoder(data)
        print(z_mean.shape)
        if z_mean.shape[1]==2:
            print("2D Plot")
            plt.figure(figsize=(12, 10), num=name)
            current_palette = sns.color_palette("deep", n_colors=np.unique(labels).shape[0])
            sns.scatterplot(z_mean[:, 0], z_mean[:, 1], hue=labels, palette=current_palette, s=100)
            # plt.colorbar()
            plt.xlabel("z[0]")
            plt.ylabel("z[1]")
            plt.show()
