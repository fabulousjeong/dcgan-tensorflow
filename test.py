import tensorflow as tf

import os
from simplegan import SIMPLEGAN
from dcgan import DCGAN

import math

import utils
from imageiteartor import ImageIterator
import datetime
import numpy as np

def test(checkpoint_dir, model, batch_size):
    
    generator = model.generator()

    checkpoint = tf.train.Checkpoint(generator=generator)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    
    noise_dim = 100
    seed = tf.random.normal([batch_size, noise_dim])
    # Produce images for the GIF as we go
    # Save to "image_at_epoch_1000.png"
    utils.generate_and_save_images(generator,1000,seed)
    


if __name__ == "__main__":
    #set hyper parameters
    batch_size = 16
    model = DCGAN()
    checkpoint_dir = './training_checkpoints'

    test(checkpoint_dir, model, batch_size)



