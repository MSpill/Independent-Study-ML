import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers

# this program generates interpolation videos from a trained GAN (varying the seed gradually so output images blend)
# Most of the code is identical to gan.py except for the last 40 lines or so

from IPython import display

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(8*5*256, use_bias=False, input_shape=(200,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((8, 5, 256)))
    assert model.output_shape == (None, 8, 5, 256) # Note: None is the batch size

    model.add(layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', output_padding=(0,0), use_bias=False))
    assert model.output_shape == (None, 15, 9, 256)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', output_padding=(1,0), use_bias=False))
    assert model.output_shape == (None, 30, 17, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', output_padding=(1,1), use_bias=False))
    assert model.output_shape == (None, 60, 34, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', output_padding=(1,1), use_bias=False))
    assert model.output_shape == (None, 120, 68, 32)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same',  output_padding=(1,0), use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 240, 135, 3)

    return model

generator = make_generator_model()

#plt.imshow(generated_image[0, :, :, 0], cmap='gray')

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[240, 135, 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(64, (5, 5), strides=(1, 1), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

discriminator = make_discriminator_model()

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = './best_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

noise_dim = 200
num_stops = 10

# We will reuse this seed overtime (so it's easier
# to visualize progress in the animated GIF)
seeds = [tf.random.normal([1, noise_dim]) for i in range(0, num_stops)]

def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = np.asarray(model(test_input, training=False))

  fig = plt.figure(figsize=(1,1))

  for i in range(predictions.shape[0]):
      plt.subplot(1, 1, i+1)
      plt.imshow((predictions[i, :, :, :].transpose(1,0,2) + 1)*0.5)
      plt.axis('off')

  plt.savefig('interpolation/image{:04d}.png'.format(epoch), dpi=300)
  #plt.show()
  plt.close()

# this is where the interpolation takes place
length = 100
for n in range(0, num_stops-1):
    for i in range(0, length):
        # use a weighted average between two samples as the seed
        # the weight slowly shifts over time
        seed = ((length-i)*seeds[n] + i*seeds[n+1])/length
        generate_and_save_images(generator, i+length*n, seed)