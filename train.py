import tensorflow as tf
tf.config.experimental_run_functions_eagerly(True)
from dcgan import DCGAN
import utils

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import matplotlib.pyplot as plt
import time

import glob
import imageio
import math


def train(train_dir, model, total_epoch, batch_size, lrate):

    # Set the image generator. we can use "ImageDataGenerator" in the keras API 
    train_image_generator = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
    train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                            directory=train_dir,
                                                            shuffle=True,
                                                            target_size=(64, 64))
    print('num:', len(train_data_gen))

    # Define the models
    generator = model.generator()
    discriminator = model.discriminator()

    # Define optimizer
    discriminator_optimizer, generator_optimizer = model.optimizer(lrate)

    # Set the ckpt to save trained weights 
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                    discriminator_optimizer=discriminator_optimizer,
                                    generator=generator,
                                    discriminator=discriminator)

    #checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


    # Set some hyper parameters 
    noise_dim = 100
    epoch_drop = 20

    # We will reuse this seed overtime (so it's easier)
    # to visualize progress in the animated GIF)
    num_examples_to_generate = 16
    seed = tf.random.normal([num_examples_to_generate, noise_dim])

    # Notice the use of `tf.function`
    # This annotation causes the function to be "compiled".
    @tf.function
    def train_step(images, discriminator_optimizer, generator_optimizer ):
        noise = tf.random.normal([batch_size, noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator(noise, training=True)

            real_output = discriminator(images, training=True)
            fake_output = discriminator(generated_images, training=True)

            disc_loss, gen_loss = model.loss(real_output, fake_output)

            gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

            generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
            discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    print ('Start Training!!!')

    for epoch in range(total_epoch):
        start = time.time()
        learning_rate = lrate * \
                math.pow(0.2, math.floor((epoch + 1) / epoch_drop))
        discriminator_optimizer, generator_optimizer = model.optimizer(learning_rate)

        #len(train_data_gen)
        for i in range(len(train_data_gen)):
            sample_training_images, _ = next(train_data_gen)
            train_step(sample_training_images, discriminator_optimizer, generator_optimizer)
            print('Training... {} / {} \r'.format(i + 1, len(train_data_gen)),end='')

        # Produce images for the GIF as we go
        utils.generate_and_save_images(generator,
                                epoch + 1,
                                seed)

        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    anim_file = 'dcgan.gif'
    utils.generated_images(anim_file)






if __name__ == "__main__":
    #set hyper parameters
    batch_size = 128
    n_noise = 100
    learning_rate = 1e-5
    total_epochs = 50

    model = DCGAN()
    
    #download align_celeba dataset from https://www.kaggle.com/jessicali9530/celeba-dataset
    #extract and move to "./data/img_align_celeba"
    train_dir = './data/img_align_celeba'

    train(train_dir, model, total_epochs, batch_size, learning_rate)



