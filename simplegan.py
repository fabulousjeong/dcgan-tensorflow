import tensorflow as tf

class SIMPLEGAN(object):
    def __init__(self, batch_size, n_noise, image_size, image_channels):
        self.batch_size = batch_size
        self.n_noise = n_noise
        self.image_size = image_size
        self.image_channels = image_channels
        self.n_input = image_size*image_size*image_channels
        self.n_hidden = 256

    
    def inputs(self):
        X = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.image_channels], name='input_sample')
        Z = tf.placeholder(tf.float32, [None, self.n_noise], name='input_noise')
        Lr = tf.placeholder(tf.float32, [], name='learning_rate')
        return X, Z, Lr
    
    #Simple generator, It consists of 2-fully connected layers. 
    def generator(self, input, reuse = False):
        with tf.variable_scope("generator") as scope:
            if reuse:
                scope.reuse_variables()
            G_W1 = tf.get_variable('G_W1', [self.n_noise, self.n_hidden], initializer = tf.random_normal_initializer(stddev=0.01))
            G_b1 = tf.get_variable('G_b1', [self.n_hidden], initializer = tf.constant_initializer(0))
            G_W2 = tf.get_variable('G_W2', [self.n_hidden, self.n_input], initializer = tf.random_normal_initializer(stddev=0.01))
            G_b2 = tf.get_variable('G_b2', [self.n_input], initializer = tf.constant_initializer(0)) 
            hidden = tf.nn.relu(
                    tf.matmul(input, G_W1) + G_b1)
            output = tf.nn.tanh(
                    tf.matmul(hidden, G_W2) + G_b2)
            output = tf.reshape(output, [self.batch_size, self.image_size, self.image_size, self.image_channels])
        return output

    # Discriminator also consists of 2-fully connected layers
    def discriminator(self, input, reuse = False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()
            D_W1 = tf.get_variable('D_W1', [self.n_input, self.n_hidden], initializer = tf.random_normal_initializer(stddev=0.01))
            D_b1 = tf.get_variable('D_b1', [self.n_hidden], initializer = tf.constant_initializer(0))
            D_W2 = tf.get_variable('D_W2', [self.n_hidden, 1], initializer = tf.random_normal_initializer(stddev=0.01))
            D_b2 = tf.get_variable('D_b2', [1], initializer = tf.constant_initializer(0))
            input = tf.reshape(input, [self.batch_size, self.n_input])
            hidden = tf.nn.relu(
                            tf.matmul(input, D_W1) + D_b1)
            output = tf.nn.sigmoid(
                            tf.matmul(hidden, D_W2) + D_b2)

        return output
    
    def loss(self, X, Z):
        g_out = self.generator(Z)
        d_fake = self.discriminator(g_out, reuse = False)
        d_real = self.discriminator(X, reuse = True)

        d_loss = tf.reduce_mean(tf.log(d_real) + tf.log(1 - d_fake))
        g_loss = tf.reduce_mean(tf.log(d_fake))
        return d_loss, g_loss

    def optimizer(self, d_loss, g_loss, learning_rate):
        d_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator')
        g_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')

        d_opt = tf.train.AdamOptimizer(learning_rate, beta1 = 0.5).minimize(-d_loss,
                                                                var_list=d_var_list)
        g_opt = tf.train.AdamOptimizer(learning_rate, beta1 = 0.5).minimize(-g_loss,
                                                                var_list=g_var_list)
        return d_opt, g_opt

    def sample(self, Z):
        g_out = self.generator(Z, reuse = True)
        return g_out


