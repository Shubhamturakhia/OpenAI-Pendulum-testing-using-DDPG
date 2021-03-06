# TODO: Defining the environment and executing the algorithm steps
# TODO: Game Engine Model and its required installation
# TODO: Game Env with execution of algorithm

import gym
import matplotlib.pyplot as plt
from Agent import *
from absl import logging,flags,app
from absl.flags import FLAGS
import time


def main(_argv):

    tn1 = time.time()
    # Environment using OpenAI gym
    env = gym.make('Pendulum-v0')
    # env.reset()
    #.seed(0)
    Episodes=100
    # Initialize agent
    Ag_controller= Agent(alpha=0.0001, beta=0.001, input_dims=[3], tau=0.001, env=env, gamma=0.99, n_act=1,
                         max_size=1000000,
                         layer1_size=400, layer2_size=300, batch_size=64)
    logging.info("Agent initialized....")
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

    reward_history =[]
    np.random.seed(0)

    for i in range(Episodes):
        logging.info("Starting episode run...")

        flag_complete = False
        s = env.reset()
        points = 0

        while not flag_complete:
            t1 = time.time()
            # This follow the steps exactly in the DDPG algorithm
            # Action chosen for the initial stage and then the parameters (SARS') are returned to be stored in buffer
            action_chosen = Ag_controller.action(s)
            new_state, reward, flag_complete, info = env.step(action_chosen)
            #print (new_state, reward, flag_complete)
            Ag_controller.get_sample_buffer(s, action_chosen, reward, new_state, int(flag_complete))

            # Learn stage
            Ag_controller.learning_stage()

            # Reward points calculation
            points = points + reward
            s = new_state
            env.render()
            t2 = time.time()
        reward_history.append(points)
        print("Episode No : {}/{} | Reward: {} | Mean Score for 100 episodes: {}".
              format(i, Episodes, float(points), np.mean(reward_history[-100:])))
        tn2 = time.time()

    plt.plot([i + 1 for i in range(0, Episodes)], reward_history)
    plt.show()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
