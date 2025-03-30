import tensorflow as tf
from mlp_autoencoder import build_mlp_autoencoder
from cnn_autoencoder import build_cnn_autoencoder
from vae_autoencoder import build_vae

# Load MNIST data
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((-1, 28, 28, 1))
x_test = x_test.reshape((-1, 28, 28, 1))

# Train MLP AutoEncoder
mlp_autoencoder = build_mlp_autoencoder(input_shape=(784,))
mlp_autoencoder.fit(x_train.reshape(-1, 784), x_train.reshape(-1, 784), epochs=10, batch_size=256, validation_data=(x_test.reshape(-1, 784), x_test.reshape(-1, 784)))

# Train CNN AutoEncoder
cnn_autoencoder = build_cnn_autoencoder(input_shape=(28, 28, 1))
cnn_autoencoder.fit(x_train, x_train, epochs=10, batch_size=256, validation_data=(x_test, x_test))

# Train VAE AutoEncoder
vae, encoder = build_vae(input_shape=(28, 28, 1))
vae.fit(x_train, epochs=10, batch_size=256, validation_data=(x_test, x_test))
