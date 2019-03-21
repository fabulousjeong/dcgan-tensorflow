import tensorflow as tf



class DCGAN(object):
    def __init__(self, batch_size, n_noise, image_size, image_channels):
        self.batch_size = batch_size
        self.n_noise = n_noise

        # but we only take 64x64
        self.image_size = image_size
        self.image_channels = image_channels
        self.n_input = image_size*image_size*image_channels
        self.n_W1 = 1024
        self.n_W2 = 512
        self.n_W3 = 256
        self.n_W4 = 128
        self.n_W5 = 64

        self.n_hidden = 4*4*self.n_W1

    # this model take input image and noise vector Z
    # can adjust learning_rate using Lr parameter
    def inputs(self):
        X = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.image_channels], name='input_sample')
        Z = tf.placeholder(tf.float32, [None, self.n_noise], name='input_noise')
        Lr = tf.placeholder(tf.float32, [], name='learning_rate')
        return X, Z, Lr
    

    # graw feature map using conv2d_transpose
    # use batch_norm and relu
    # output size is 64x64
    def generator(self, input, reuse = False):
        with tf.variable_scope("generator") as scope:
            if reuse:
                scope.reuse_variables()
            G_FW1 = tf.get_variable('G_FW1', [self.n_noise, self.n_hidden], initializer = tf.random_normal_initializer(stddev=0.01))
            G_Fb1 = tf.get_variable('G_Fb1', [self.n_hidden], initializer = tf.constant_initializer(0))

            G_W1 = tf.get_variable('G_W1', [5,5,self.n_W2, self.n_W1], initializer = tf.truncated_normal_initializer(stddev=0.02))
            G_W2 = tf.get_variable('G_W2', [5,5,self.n_W3, self.n_W2], initializer = tf.truncated_normal_initializer(stddev=0.02))
            G_W3 = tf.get_variable('G_W3', [5,5,self.n_W4, self.n_W3], initializer = tf.truncated_normal_initializer(stddev=0.02))
            G_W4 = tf.get_variable('G_W4', [5,5,self.image_channels, self.n_W4], initializer = tf.truncated_normal_initializer(stddev=0.02))

        hidden = tf.nn.relu(
                tf.matmul(input, G_FW1) + G_Fb1)
        hidden = tf.reshape(hidden, [self.batch_size, 4,4,self.n_W1]) 
        dconv1 = tf.nn.conv2d_transpose(hidden, G_W1, [self.batch_size, 8, 8, self.n_W2], [1, 2, 2, 1])
        dconv1 = tf.nn.relu(tf.contrib.layers.batch_norm(dconv1,decay=0.9, epsilon=1e-5))

        dconv2 = tf.nn.conv2d_transpose(dconv1, G_W2, [self.batch_size, 16, 16, self.n_W3], [1, 2, 2, 1])
        dconv2 = tf.nn.relu(tf.contrib.layers.batch_norm(dconv2,decay=0.9, epsilon=1e-5))

        dconv3 = tf.nn.conv2d_transpose(dconv2, G_W3, [self.batch_size, 32, 32, self.n_W4], [1, 2, 2, 1])
        dconv3 = tf.nn.relu(tf.contrib.layers.batch_norm(dconv3,decay=0.9, epsilon=1e-5))

        dconv4 = tf.nn.conv2d_transpose(dconv3, G_W4, [self.batch_size, 64, 64, self.image_channels], [1, 2, 2, 1])

        output = tf.nn.tanh(dconv4)
        return output

    # symmetrical structure with the generator
    # use conv2d with stride size 2. 
    # Same as in generator, batchnorm is placed at the end of each layer. 
    # But leaky relu is used as the activate function.
    def discriminator(self, input, reuse = False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()


            D_W1 = tf.get_variable('D_W1', [5,5,self.image_channels, self.n_W5], initializer = tf.truncated_normal_initializer(stddev=0.02))
            D_W2 = tf.get_variable('D_W2', [5,5,self.n_W5, self.n_W4], initializer = tf.truncated_normal_initializer(stddev=0.02))
            D_W3 = tf.get_variable('D_W3', [5,5,self.n_W4, self.n_W3], initializer = tf.truncated_normal_initializer(stddev=0.02))
            D_W4 = tf.get_variable('D_W4', [5,5,self.n_W3, self.n_W2], initializer = tf.truncated_normal_initializer(stddev=0.02)) 

            D_FW1 = tf.get_variable('D_FW1', [4*4*self.n_W2, 1], initializer = tf.random_normal_initializer(stddev=0.01))
            D_Fb1 = tf.get_variable('D_Fb1', [1], initializer = tf.constant_initializer(0))


        conv1 = tf.nn.conv2d(input, D_W1, strides = [1, 2, 2, 1], padding='SAME')
        conv1 = tf.nn.leaky_relu(conv1, alpha = 0.2)

        conv2 = tf.nn.conv2d(conv1, D_W2, strides = [1, 2, 2, 1], padding='SAME')
        conv2 = tf.nn.leaky_relu(tf.contrib.layers.batch_norm(conv2, decay=0.9, epsilon=1e-5), alpha = 0.2)

        conv3 = tf.nn.conv2d(conv2, D_W3, strides = [1, 2, 2, 1], padding='SAME')
        conv3 = tf.nn.leaky_relu(tf.contrib.layers.batch_norm(conv3, decay=0.9, epsilon=1e-5), alpha = 0.2)

        conv4 = tf.nn.conv2d(conv3, D_W4, strides = [1, 2, 2, 1], padding='SAME')
        conv4 = tf.nn.leaky_relu(tf.contrib.layers.batch_norm(conv4, decay=0.9, epsilon=1e-5), alpha = 0.2)

        hidden = tf.reshape(conv4, [self.batch_size, 4*4*self.n_W2]) 

        output = tf.nn.sigmoid(
                        tf.matmul(hidden, D_FW1) + D_Fb1)

        return output
    
    # Loss function and optimizer are same as simple GAN
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
        #print('G_var_list:', len(G_var_list))
        #print('D_var_list:', len(D_var_list))

        d_opt = tf.train.AdamOptimizer(learning_rate, beta1 = 0.5).minimize(-d_loss,
                                                                var_list=d_var_list)
        g_opt = tf.train.AdamOptimizer(learning_rate, beta1 = 0.5).minimize(-g_loss,
                                                                var_list=g_var_list)
        return d_opt, g_opt

    def sample(self, Z, reuse = True):
        g_out = self.generator(Z, reuse = reuse)
        return g_out

