import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.compat.v1.initializers import random_uniform

tf.compat.v1.disable_eager_execution()


class ActorNN(object):

    def __init__(self, sess, learning_rate, n_act ,input_dims, name, layer1_dims, layer2_dims,
                 action_bound, batch_size=64, ckpt = "DDpg_checkpoints"):
        self.sess = sess
        self.learning_rate = learning_rate
        self.input_dims = input_dims
        print(self.input_dims)
        self.name = name
        self.batch_size = batch_size
        self.f1 = layer1_dims
        self.f2 = layer2_dims
        self.action_bound = action_bound
        self.ckpt = ckpt
        self.n_act = n_act
        print ("ACTOR NETWORK")
        # Need to build the network at Initialization, Save the checkpoints in the ckpt directory named above
        self.create_actor_network()
        self.network_parameters =  tf.compat.v1.trainable_variables(scope = self.name)
        self.save = tf.compat.v1.train.Saver()
        self.checkpoint_file = os.path.join(ckpt, name+'_ddpg')

        self.gradients1_unnorm = tf.gradients(self.mu,self.network_parameters,self.actor_gradient)

        self.actor_gradients = list(map(lambda x: tf.compat.v1.div(x, self.batch_size), self.gradients1_unnorm))

        self.optimizer = Adam(self.learning_rate).\
            apply_gradients(zip(self.actor_gradients, self.network_parameters))

    def create_actor_network(self):
            with tf.compat.v1.variable_scope(self.name, reuse=tf.compat.v1.AUTO_REUSE):

                self.input = tf.compat.v1.placeholder(tf.float32, shape=[None, *self.input_dims],
                             name='inputs')

                self.actor_gradient = tf.compat.v1.placeholder(tf.float32,
                                      shape=[None, self.n_act],
                                      name='gradient')

                # Defining the layers: Layer 1: normal dense, batch_norm and activation= relu -- given 400
                #                      Layer 2: normal dense, batch_norm and activation= relu -- given 300
                #                      Final Layer: Tanh, batch_norm and activation= relu

                s1 = 1. / np.sqrt(self.f1)
                weights = random_uniform(-s1, s1)
                bias = random_uniform(-s1, s1)
                Layer1 = tf.compat.v1.layers.dense(self.input, units=self.f1,
                                               kernel_initializer=weights,
                                               bias_initializer=bias)
                Norm1 = tf.compat.v1.layers.batch_normalization(Layer1)
                L1_Activation = tf.keras.activations.relu(Norm1)

                s2 = 1. / np.sqrt(self.f2)
                weights = random_uniform(-s2, s2)
                bias = random_uniform(-s2, s2)
                Layer2 = tf.compat.v1.layers.dense(L1_Activation, units=self.f2, kernel_initializer=weights,
                                               bias_initializer=bias)
                Norm2 = tf.compat.v1.layers.batch_normalization(Layer2)
                L2_Activation = tf.keras.activations.relu(Norm2)

                s3 = 0.003
                weights = random_uniform(-s3, s3)
                bias = random_uniform(-s3, s3)
                Layer3 = tf.compat.v1.layers.dense(L2_Activation, units=self.n_act, kernel_initializer=weights,
                                               bias_initializer=bias)
                mu = tf.keras.activations.tanh(Layer3)
                print ("STAGING TO MU")
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