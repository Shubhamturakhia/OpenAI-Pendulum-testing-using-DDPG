# TODO: Make sure the parameters for the agent are initialized in the Initialize function
# TODO: Make sure the updation of actor anfd critic target networks are done
"""
  TODO: 1. Function to get sample buffer
        2. Function to do the Learning process
        3. Function
"""
from Resources import OUNoise, ReplayBuffer
from Actor_Network import *
from Critic_Network import *


class Agent(object):

    def __init__(self, alpha, beta, input_dims, tau, env, gamma=0.99, n_act=1,
                 max_size=10e6, layer1_size=400, layer2_size=300,
                 batch_size=64):
        self.gamma = gamma
        self.tau = tau
        self.RB = ReplayBuffer(max_size, input_dims, n_act)
        self.batch_size = batch_size
        self.sess = tf.compat.v1.Session()

        self.noise = OUNoise(mu=np.zeros(n_act))
        print("ACTOR NETWORK AGENT")
        self.actor = ActorNN(learning_rate=alpha, input_dims=input_dims, name='Actor', sess=self.sess, n_act=n_act,
                             layer1_dims=layer1_size, layer2_dims=layer2_size,action_bound=env.action_space.high)
        print("CRTIC NETWORK AGENT")
        self.critic = CriticNN(learning_rate=beta, input_dims=input_dims, name='Critic', sess=self.sess, n_act=n_act,
                               layer1_dims=layer1_size, layer2_dims=layer2_size)

        self.target_a = ActorNN(learning_rate=alpha, n_act=n_act, name='TargetActor',
                                input_dims=input_dims, sess=self.sess, layer1_dims=layer1_size,
                                layer2_dims=layer2_size, action_bound=env.action_space.high)
        self.target_c = CriticNN(learning_rate=beta, n_act=n_act, name='TargetCritic', input_dims=input_dims,
                                 sess=self.sess, layer1_dims=layer1_size, layer2_dims=layer2_size)

        self.update_critic = [self.target_c.network_parameters[i].assign(
            tf.multiply(self.critic.network_parameters[i], self.tau) +
            tf.multiply(self.target_c.network_parameters[i], 1. - self.tau))
            for i in range(len(self.target_c.network_parameters))]

        self.update_actor = [self.target_a.network_parameters[i].assign(
            tf.multiply(self.actor.network_parameters[i], self.tau) +
            tf.multiply(self.target_a.network_parameters[i], 1. - self.tau))
            for i in range(len(self.target_a.network_parameters))]

        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.update_parameters()

    def learning_stage(self):
        if self.RB.memory_cntr < self.batch_size:
            return

        state, action, reward, new_state, flag_complete = self.RB.sample_buffer(self.batch_size)

        updated_critic_value = self.target_c.predict(new_state,
                                                          self.target_a.predict(new_state))
        yi = []
        for k in range(self.batch_size):
            yi.append(reward[k] + self.gamma * updated_critic_value[k] * flag_complete[k])
        yi = np.reshape(yi, (self.batch_size, 1))

        _ = self.critic.train(state, action, yi)

        action_outputs = self.actor.predict(state)
        grad = self.critic.get_action_gradients(state, action_outputs)

        self.actor.train(state, grad[0])
        self.update_parameters()

    def get_sample_buffer(self, state, action, reward, new_state, flag_complete):
        #print(flag_complete)
        return self.RB.transition(state, action, reward, new_state, flag_complete)

    def update_parameters(self):
        initial_session = False

        if initial_session:
            old_tau = self.tau
            self.tau = 1.0
            self.target_c.sess.run(self.update_critic)
            self.target_a.sess.run(self.update_actor)
            self.tau = old_tau
        else:
            self.target_c.sess.run(self.update_critic)
            self.target_a.sess.run(self.update_actor)

    def action(self, state):
        state = state[np.newaxis, :]
        mu = self.actor.predict(state)
        noise = self.noise()
        mu_dash = mu + noise
        return mu_dash[0]