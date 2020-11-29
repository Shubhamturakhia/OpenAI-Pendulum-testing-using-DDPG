import os
import numpy as np
import tensorflow as tf
from tensorflow.compat.v1.initializers import random_uniform

# TODO: Creation of Critic Network
# TODO: Minimize loss function
# TODO: Creation of checkpoints, saving them,(add best checkpoints according to the rewards achieved and no deletion or resetting for that - until mentioned or confirmed)
# TODO: Actor gradients

tf.compat.v1.disable_eager_execution()
class CriticNN(object):

    def __init__(self, sess, learning_rate, n_act, input_dims, name, layer1_dims, layer2_dims,
                 batch_size=64, ckpt="DDpg_checkpoints"):
        self.sess = sess
        self.learning_rate = learning_rate
        self.input_dims = input_dims
        self.name = name
        self.batch_size = batch_size
        self.f1 = layer1_dims
        self.f2 = layer2_dims
        self.ckpt = ckpt
        self.n_act = n_act
        print("CRITIC NETWORK")
        # Need to build the network at Initialization, Save the checkpoints in the ckpt directory named above
        self.create_critic_network()
        self.network_parameters = tf.compat.v1.trainable_variables(scope=self.name)
        self.save = tf.compat.v1.train.Saver()
        self.checkpoint_file = os.path.join(ckpt, name + '_ddpg')

        # minimize the loss function as is in the paper
        self.actor_gradients = tf.gradients(self.q, self.actions)

        self.optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def create_critic_network(self):
            with tf.compat.v1.variable_scope(self.name):

                # adding placeholders
                self.input = tf.compat.v1.placeholder(tf.float32, shape=[None, *self.input_dims],
                                            name='inputs')

                self.actions = tf.compat.v1.placeholder(tf.float32, shape=[None, self.n_act],
                                              name='actions')

                self.targetq = tf.compat.v1.placeholder(tf.float32, shape=[None, 1], name='target_q')

                # Defining the layers: Layer 1: normal dense, batch_norm and activation= relu -- given 400
                #                      Layer 2: normal dense, batch_norm and activation= relu -- given 300
                #                      Combining the states and actions
                #                      Finding the Q values
                #                      Finding the loss

                s1 = 1. / np.sqrt(self.f1)
                weights = random_uniform(-s1, s1)
                bias = random_uniform(-s1, s1)
                Layer1 = tf.compat.v1.layers.dense(self.input, units=self.f1, kernel_initializer=weights,
                                               bias_initializer=bias)
                Norm1 = tf.compat.v1.layers.batch_normalization(Layer1)
                L1_Activation = tf.keras.activations.relu(Norm1)

                s2 = 1. / np.sqrt(self.f2)
                weights = random_uniform(-s2, s2)
                bias = random_uniform(-s2, s2)
                Layer2 = tf.compat.v1.layers.dense(L1_Activation, units=self.f2, kernel_initializer=weights,
                                               bias_initializer=bias)
                Norm2 = tf.compat.v1.layers.batch_normalization(Layer2)
                action_in = tf.compat.v1.layers.dense(self.actions, units=self.f2,
                                                  activation='relu')

                state_actions = tf.add(Norm2, action_in)
                state_actions = tf.keras.activations.relu(state_actions)

                s3 = 3e-3
                weights = random_uniform(-s3, s3)
                bias = random_uniform(-s3, s3)
                self.q = tf.compat.v1.layers.dense(state_actions, units=1, kernel_initializer=weights,
                                               bias_initializer=bias)
                print("STAGING TO LOSSS")
                self.loss = tf.keras.metrics.mean_squared_error(self.targetq, self.q)

    def predict(self, inputs, actions):
        return self.sess.run(self.q, feed_dict={self.input: inputs,
                             self.actions: actions})

    def train(self, inputs, actions, targetq):
        return self.sess.run(self.optimizer, feed_dict={self.input: inputs,
                             self.actions: actions,
                             self.targetq: targetq})

    def get_action_gradients(self, inputs, actions):
        return self.sess.run(self.actor_gradients,
                             feed_dict={self.input: inputs,
                                        self.actions: actions})

    def load_checkpoint(self):
        print("## Loading checkpoint ##")
        self.save.restore(self.sess, self.checkpoint_file)

    def save_checkpoint(self):
        print("## Saving checkpoint ##")
        self.save.save(self.sess, self.checkpoint_file)
