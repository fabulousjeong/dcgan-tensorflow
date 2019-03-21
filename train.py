import tensorflow as tf

import os
from simplegan import SIMPLEGAN
from dcgan import DCGAN

import math

import utils
from imageiteartor import ImageIterator


def train(data_root, model, total_epoch, batch_size, lrate):

    X, Z, Lr = model.inputs()
    d_loss, g_loss = model.loss(X, Z)
    d_opt, g_opt = model.optimizer(d_loss, g_loss, Lr)
    g_sample = model.sample(Z)
    sample_size = batch_size
    test_noise = utils.get_noise(sample_size, n_noise)
    epoch_drop = 3

    iterator, image_count = ImageIterator(data_root, batch_size, model.image_size, model.image_channels).get_iterator()
    next_element = iterator.get_next()

    total_batch = int(image_count/batch_size)
    #learning_rate = lrate
    #G_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(iterator.initializer)

        for epoch in range(total_epoch):
            learning_rate = lrate * \
                            math.pow(0.2, math.floor((epoch + 1) / epoch_drop))
            for step in range(total_batch):
                batch_x = sess.run(next_element)
                batch_z = utils.get_noise(batch_size, n_noise)

                _, loss_val_D = sess.run([d_opt, d_loss],
                                        feed_dict={X: batch_x, Z: batch_z, Lr: learning_rate})
                _, loss_val_G = sess.run([g_opt, g_loss],
                                        feed_dict={Z: batch_z, Lr: learning_rate})

                if step % 300 == 0:
                    #sample_size = 10
                    #noise = get_noise(sample_size, n_noise)
                    samples = sess.run(g_sample, feed_dict={Z: test_noise})
                    title = 'samples/%05d_%05d.png'%(epoch, step)
                    utils.save_samples(title, samples)
                    
                    print('Epoch:', '%04d' % epoch,
                    '%05d/%05d' % (step, total_batch),
                    'D loss: {:.4}'.format(loss_val_D),
                    'G loss: {:.4}'.format(loss_val_G))
            saver.save(sess, './models/dcgan', global_step=epoch)






if __name__ == "__main__":
    #set hyper parameters
    batch_size = 32
    n_noise = 100
    image_size = 64
    image_channels = 3
    learning_rate = 0.0002
    total_epochs = 20

    #model = SIMPLEGAN(batch_size, n_noise, image_size, image_channels)
    model = DCGAN(batch_size, n_noise, image_size, image_channels)
    #data_root = '../data/mnist/trainingSet' 
    
    #download align_celeba dataset from https://www.kaggle.com/jessicali9530/celeba-dataset
    #extract and move to "./data/img_align_celeba"
    data_root = './data/img_align_celeba'

    with tf.Graph().as_default():
        train(data_root, model, total_epochs, batch_size, learning_rate)



