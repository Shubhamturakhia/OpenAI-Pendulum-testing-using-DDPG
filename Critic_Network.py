import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.compat.v1.initializers import random_uniform


class CriticNN(object):

    def __init__(self, sess, learning_rate, n_act, input_dims, name, batch_size, layer1_dims, layer2_dims,
                 action_bound, ckpt = "DDpg_checkpoints"):
        self.sess = sess
        self.learning_rate = learning_rate
        self.input_dims = input_dims
        self.name = name
        self.batch_size = batch_size
        self.f1 = layer1_dims
        self.f2 = layer2_dims
        self.action_bound = action_bound
        self.ckpt = ckpt
        self.n_act = n_act

        # Need to build the network at Initialization, Save the checkpoints in the ckpt directory named above
        self.create_critic_network()
        self.network_parameters = tf.compat.v1.trainable_variables(scope = self.name)
        self.save = tf.compat.v1.train.Saver()
        self.checkpoint_file = os.path.join(ckpt, name+'_ddpg')

        # minimize the loss function as is in the paper
        self.actor_gradients = tf.gradients(self.q, self.actions)

        self.optimizer = Adam(self.learning_rate).minimize(self.loss)

    def create_critic_network(self):
        while True:
            with tf.compat.v1.variable_scope(self.name):

                self.input = tf.keras.Input(tf.float32, shape=[None, *self.input_dims],
                                            name='inputs')

                self.actions = tf.keras.Input(tf.float32, shape=[None, self.n_act],
                                                     name='actions')

                self. targetq - tf.keras.Input(tf.float32, shape=[None, 1],name ='target_q')

                # Defining the layers: Layer 1: normal dense, batch_norm and activation= relu -- given 400
                #                      Layer 2: normal dense, batch_norm and activation= relu -- given 300
                #                      Combining the states and actions
                #                      Finding the Q values
                #                      Finding the loss

                s1 = 1. / np.sqrt(self.f1)
                weights = random_uniform(-s1, s1)
                bias = random_uniform(-s1, s1)
                Layer1 = tf.keras.layers.Dense(self.input, units=self.f1,kernel_initializer=weights,
                                               bias_initializer=bias)
                Norm1 = tf.keras.layers.BatchNormalization(Layer1)
                L1_Activation = tf.keras.activations.relu(Norm1)

                s2 = 1. / np.sqrt(self.f2)
                weights = random_uniform(-s2, s2)
                bias = random_uniform(-s2, s2)
                Layer2 = tf.keras.layers.Dense(L1_Activation, units=self.f2, kernel_initializer=weights,
                                               bias_initializer=bias)
                Norm2 = tf.keras.layers.BatchNormalization(Layer2)
                L2_Activation = tf.keras.activations.relu(Norm2)

                s3 = 3e-3
                weights = random_uniform(-s3, s3)
                bias = random_uniform(-s3, s3)
                Layer3 = tf.keras.layers.Dense(L2_Activation, units=self.n_act, kernel_initializer=weights,
                                               bias_initializer=bias)
                mu = tf.keras.activations.tanh(Layer3)

                self.mu = tf.multiply(mu, self.action_bound)

    def predict(self, inputs):
        return self.sess.run(self.mu, feed_dict={self.input: inputs})

    def train(self, inputs, gradients):
        self.sess.run(self.optimizer, feed_dict={self.input: inputs, self.actor_gradient: gradients})

    def load_checkpoint(self):
        print("## Loading checkpoint ##")
        self.save.restore(self.sess, self.checkpoint_file)

    def save_checkpoint(self):
        print("## Saving checkpoint ##")
        self.save.save(self.sess, self.checkpoint_file)



