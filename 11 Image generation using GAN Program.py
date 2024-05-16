import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers

# Define the Generator Model
def build_generator(latent_dim):
  """
  Creates a sequential model for generating images from latent space.

  Args:
      latent_dim: Dimensionality of the latent space.

  Returns:
      A TensorFlow Keras sequential model.
  """
  model = tf.keras.Sequential()
  model.add(layers.Dense(128, input_dim=latent_dim, activation='relu'))
  model.add(layers.BatchNormalization())
  model.add(layers.Dense(784, activation='sigmoid'))
  model.add(layers.Reshape((28, 28, 1)))
  return model

# Define the Discriminator Model
def build_discriminator(img_shape):
  """
  Creates a sequential model for classifying real and generated images.

  Args:
      img_shape: Shape of the input image (e.g., (28, 28, 1) for MNIST).

  Returns:
      A TensorFlow Keras sequential model.
  """
  model = tf.keras.Sequential()
  model.add(layers.Flatten(input_shape=img_shape))
  model.add(layers.Dense(128, activation='relu'))
  model.add(layers.Dense(1, activation='sigmoid'))
  return model

# Define the Combined GAN Model (Generator + Discriminator)
def build_gan(generator, discriminator):
  """
  Combines the generator and discriminator models for training the GAN.

  Args:
      generator: The generator model.
      discriminator: The discriminator model.

  Returns:
      A TensorFlow Keras sequential model (combined GAN).
  """
  discriminator.trainable = False  # Freeze discriminator during combined training
  model = tf.keras.Sequential()
  model.add(generator)
  model.add(discriminator)
  return model

# Load and Preprocess the Dataset (MNIST in this example)
def load_dataset():
  """
  Loads the MNIST dataset and preprocesses it for the GAN.

  Returns:
      A NumPy array containing the preprocessed training images.
  """
  (X_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
  X_train = X_train / 255.0  # Normalize pixel values to [0, 1]
  X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)  # Reshape for CNN input
  return X_train

# Train the Generative Adversarial Network (GAN)
def train_gan(generator, discriminator, gan, X_train, latent_dim, epochs=10000, batch_size=128):
  """
  Trains the GAN by alternately training the discriminator and generator.

  Args:
      generator: The generator model.
      discriminator: The discriminator model.
      gan: The combined GAN model.
      X_train: The training data (preprocessed images).
      latent_dim: Dimensionality of the latent space.
      epochs: Number of training epochs (default: 10000).
      batch_size: Training batch size (default: 128).
  """
  for epoch in range(epochs):

    # Train the Discriminator
    idx = np.random.randint(0, X_train.shape[0], batch_size)
    real_imgs = X_train[idx]
    fake_imgs = generator.predict(np.random.randn(batch_size, latent_dim))
    labels_real = np.ones((batch_size, 1))
    labels_fake = np.zeros((batch_size, 1))
    d_loss_real = discriminator.train_on_batch(real_imgs, labels_real)
    d_loss_fake = discriminator.train_on_batch(fake_imgs, labels_fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # Train the Generator (via the combined model)
    noise = np.random.randn(batch_size, latent_dim)
    labels_gen = np.ones((batch_size, 1))
    g_loss = gan.train_on_batch(
