# TODO: Defining the environment and executing the algorithm steps
# TODO: Game Engine Model and its required installation
# TODO: Game Env with execution of algorithm

import gym
import matplotlib.pyplot as plt
from Agent import *

if __name__ == "__main__":

    # Environment using OpenAI gym
    env =gym.make('Pendulum-v0')
    # env.reset()
    env.seed(0)
    Episodes=500
    # Initialize agent
    Ag_controller= Agent(alpha=10e-4, beta=10e-3, input_dims=[3], tau= 0.001, env=env, gamma=0.99, n_act=1,
                         max_size=10e6,
                         layer1_size=400, layer2_size=300, batch_size=64)
    """
    alpha = 10e-4
    beta = 10e-3
    input_dims = [3]
    tau = 0.001
    gamma = 0.99
    n_act = 1,
    max_size = 10e6,
    layer1_size = 400
    layer2_size = 300
    """

    np.random.seed(0)
    reward_history =[]

    for i in range(Episodes):
        print("Starting....")

        flag_complete = False
        s = env.reset()
        points = 0

        while not flag_complete:
            env.render()
            # This follow the steps exactly in the DDPG algorithm
            # Action chosen for the initial stage and then the parameters (SARS') are returned to be stored in buffer
            action_chosen = Ag_controller.action(s)
            new_state, reward, done, info = env.step(action_chosen)
            k = Ag_controller.get_sample_buffer(s, action_chosen, reward, new_state, flag_complete)

            #Learn stage
            def learning_stage(self):
                if self.memory.mem_cntr < self.batch_size:
                    return

                state, action, reward, new_state, flag_complete = self.memory.sample_buffer(self.batch_size)

                updated_critic_value = self.target_critic.predict(new_state,
                                                           self.target_actor.predict(new_state))
                yi = []
                for k in range(self.batch_size):
                    yi.append(reward[k] + self.gamma * updated_critic_value[k] * flag_complete[k])
                yi = np.reshape(yi, (self.batch_size, 1))

                predicted_q, _ = self.critic.train(state, action, yi)

                action_outputs = self.actor.predict(state)
                grad = self.critic.get_action_gradients(state, action_outputs)

                self.actor.train(state, grad[0])

                self.update_parameters()

            points = points + reward
            s = new_state

        reward_history.append(points)

        print("Episode No : {}/{} | Reward: {} | Mean Score for 100 episodes: {}".
              format(i, Episodes, float(points), np.mean(reward_history[-100:])))

        plt.plot([i + 1 for i in range(0, Episodes)], points)
        plt.show()