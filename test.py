import tensorflow as tf

import os
from simplegan import SIMPLEGAN
from dcgan import DCGAN

import math

import utils
from imageiteartor import ImageIterator
import datetime
import numpy as np

def test(ckpt_root, model, batch_size):

    X, Z, Lr = model.inputs()
    g_sample = model.sample(Z, reuse = False)
    sample_size = batch_size
    test_noise = utils.get_noise(sample_size, n_noise)

    # TEST DCGAN interpolation
    '''
    test_noise[0] = np.array([[-0.01403, 0.56896, 1.41881, 0.02516, -1.36731, -0.92614, 1.17105, -0.17130, -0.11242, 0.35453, 0.06243, 0.41190, -0.18923, -1.10846, -0.10500, 0.65989, -0.19307, -0.32606, -1.77017, -0.38637, -0.82117, 0.53288, -0.38393, 1.16999, 0.02266, 0.36757, 0.13555, -1.06630, 0.00951, -0.04134, -0.29982, -0.83991, -0.04059, -0.56064, 0.39640, 0.29686, 0.42023, -1.15875, 0.19443, -0.89730, 0.37836, -2.48704, -0.03874, 0.04086, -0.35425, -0.02359, 0.56843, -0.45289, 1.79295, 0.98343, -0.99543, 0.70134, -1.43882, -0.10630, -0.39800, -1.90689, -0.16606, 0.01075, 0.11386, 0.08757, 0.25799, 1.06645, 0.07529, 1.17719, 1.38717, -0.93715, 0.60258, 0.64817, -0.70972, 1.49177, -0.58564, -1.47612, -0.49625, 2.30098, -0.08210, -0.22495, -0.47805, -0.72601, 0.58665, -0.63158, 0.04414, -0.05951, -0.92667, -0.07905, -2.26017, -0.29677, 0.93230, -0.06546, -0.46701, 1.49024, 0.01060, -0.86621, -0.65857, 0.42297, -1.43760, 0.53813, -0.13808, 0.23095, -0.78151, 0.63207
]])
    test_noise[8] = np.array([[-0.30745, -1.80994, 0.84740, -0.01723, -0.25759, 1.62209, -0.01877, 1.31540, 1.80470, -1.76964, 2.06064, -0.62803, -0.94382, 0.85376, -0.26913, 0.69890, 1.52500, -0.62958, -0.97269, 1.81976, 1.46848, -0.10180, -0.14649, 0.82289, -0.21654, 0.63229, -0.61106, -0.84134, 0.95145, -0.84128, -0.02509, -0.14419, -0.46364, -0.00298, -0.23900, -1.37273, -1.16797, 1.10777, -1.56686, -0.60846, -0.18123, 1.95980, 0.58466, 0.64532, 1.01655, -1.00187, 0.07544, 0.31779, 1.55344, -0.41186, -0.14158, 0.07359, -1.02670, -0.14173, 0.10773, -0.64202, -1.56408, -1.96202, -0.13097, -0.05426, -2.26692, 0.04790, 0.03724, -0.55998, 0.11415, -1.97006, 0.56635, -1.29249, 0.32449, 0.37213, -0.77510, -0.09502, 2.44859, 0.68632, 0.48752, 0.18134, -1.14473, -0.09552, -0.62953, 0.28095, -1.04062, 0.39957, -1.39301, -0.29697, -0.99899, 1.91437, -1.94361, -0.38661, -0.04163, -0.09743, -0.87291, 1.00404, 0.51789, -0.78019, 1.43526, 0.16111, 1.26596, -0.12284, 0.74221, 1.53793
]])
    test_noise[1] = (7*test_noise[0]+1*test_noise[8])/8
    test_noise[2] = (6*test_noise[0]+2*test_noise[8])/8
    test_noise[3] = (5*test_noise[0]+3*test_noise[8])/8
    test_noise[4] = (4*test_noise[0]+4*test_noise[8])/8
    test_noise[5] = (3*test_noise[0]+5*test_noise[8])/8
    test_noise[6] = (2*test_noise[0]+6*test_noise[8])/8
    test_noise[7] = (1*test_noise[0]+7*test_noise[8])/8
    '''



    saver = tf.train.Saver()

    with tf.Session() as sess:
        #sess.run(tf.global_variables_initializer())
        saver.restore(sess, ckpt_root)

        samples = sess.run(g_sample, feed_dict={Z: test_noise})
        date = datetime.datetime.now()
        title = 'samples/%s.png' % date
        utils.save_samples(title, samples)
    np.savetxt('samples/test_noise%s.txt' % date, test_noise, fmt='%2.5f', delimiter=', ')


    


if __name__ == "__main__":
    #set hyper parameters
    batch_size = 9
    n_noise = 100
    image_size = 64
    image_channels = 3
    learning_rate = 0.0002
    total_epochs = 20

    tf.reset_default_graph()

    #model = SIMPLEGAN(batch_size, n_noise, image_size, image_channels)
    model = DCGAN(batch_size, n_noise, image_size, image_channels)

    ckpt_root = './models/dcgan-16'

    with tf.Graph().as_default():
        test(ckpt_root, model, batch_size)



