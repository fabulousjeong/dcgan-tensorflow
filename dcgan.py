from tensorflow.keras import layers
import tensorflow as tf


class DCGAN(object):
    def __init__(self):
        return

    # graw feature map using conv2d_transpose
    # use batch_norm and relu
    # output size is 64x64
    def generator(self):
        model = tf.keras.Sequential()
        model.add(layers.Dense(4*4*1024, use_bias=False, input_shape=(100,)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Reshape((4, 4, 1024)))
        assert model.output_shape == (None, 4, 4, 1024) # Note: None is the batch size

        model.add(layers.Conv2DTranspose(512, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        assert model.output_shape == (None, 4, 4, 512)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 8, 8, 256)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 16, 16, 128)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 32, 32, 64)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='sigmoid'))
        assert model.output_shape == (None, 64, 64, 3)

        return model

    # symmetrical structure with the generator
    # use conv2d with stride size 2. 
    # Same as in generator, batchnorm is placed at the end of each layer. 
    # But leaky relu is used as the activate function.
    def discriminator(self):
        model = tf.keras.Sequential()
        model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                        input_shape=[64, 64, 3]))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Flatten())
        model.add(layers.Dense(1))

        return model
    
    # Loss function and optimizer are same as simple GAN
    def loss(self, real_output, fake_output):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        d_loss = real_loss + fake_loss
        g_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
        return d_loss, g_loss

    def optimizer(self, learning_rate):
        d_opt = tf.keras.optimizers.Adam(learning_rate)
        g_opt = tf.keras.optimizers.Adam(learning_rate)
        return d_opt, g_opt
