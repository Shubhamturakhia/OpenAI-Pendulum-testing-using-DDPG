# TODO: Make sure the parameters for the agent are initialized in the Initialize function
# TODO: Make sure the updation of actor anfd critic target networks are done
"""
  TODO: 1. Function to get sample buffer
        2. Function to do the Learning process
        3. Function
"""
import tensorflow as tf
from Resources import OUNoise,ReplayBuffer
from Actor_Network import *
from Critic_Network import *


class Agent(object):

    def __init__(self, alpha, beta, input_dims, tau, env, gamma=0.99, n_act=2,
                 max_size=1000000, layer1_size=400, layer2_size=300,
                 batch_size=64):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_act)
        self.batch_size = batch_size
        self.sess = tf.compat.v1.Session()

        self.noise = OUNoise(mu=np.zeros(n_act))
        self.actor = ActorNN(alpha, n_act, 'Actor', input_dims, self.sess,
                           layer1_size, layer2_size, env.action_space.high)
        self.critic = CriticNN(beta, n_act, 'Critic', input_dims,self.sess,
                             layer1_size, layer2_size)

        self.target_a = ActorNN(alpha, n_act, 'TargetActor',
                                  input_dims, self.sess, layer1_size,
                                  layer2_size, env.action_space.high)
        self.target_c = CriticNN(beta, n_act, 'TargetCritic', input_dims,
                                      self.sess, layer1_size, layer2_size)

        self.update_critic = [self.target_c.network_parameters[i].assign(
                      tf.multiply(self.critic.network_parameters[i], self.tau) +
                      tf.multiply(self.target_c.network_parameters[i], 1. - self.tau))
         for i in range(len(self.target_c.network_parameters))]

        self.update_actor = [self.target_a.network_parameters[i].assign(
                      tf.multiply(self.actor.network_parameters[i], self.tau) +
                      tf.multiply(self.target_a.network_parameters[i], 1. - self.tau))
         for i in range(len(self.target_a.network_parameters))]

        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.update_network_parameters(first=True)

