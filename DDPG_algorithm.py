# Masters Thesis: DDPG algorithm RL technique for Bike Control and Navigation
# Fall 2020
"""
ALGORITHM DESIGN:

Following the DDPG and Deep Q Learning algorithm procedure, we need the old state, action, reward and new states as
output for every step

Things required for Algorithm processing:
1. Class for Storing the previous values (rewards and states) -
2. Class for Actor DNN : class Actor(Obj)
3. Class for Critic DNN: class Critic(Obj)
4. Class for Ornstein Unlenbeck (this would be a class defined for Noise) - Ornstein-Uhlenbeck process (Uhlenbeck &
Ornstein, 1930) - page 4 - (https://arxiv.org/pdf/1509.02pdf971.)
5. Class or function for Memory size and/or terminal size
6. Loading the environment and deployment of Algorithm.
7. Class for Agent to do the learning  and make use of the classes above: class Agent(Obj)

Constraints required:
1. Deterministic policy is action based (i.e it O/P is "Action" and Not a "Prob value")
2. Limiting the constraints in the environment designed
"""

# Import the libraries required for processing
import os
import tensorflow as tf
from tensorflow.keras import regularizers, optimizers, activations, layers
import numpy as np
from absl import app, flags

# This class is to define the noise (Ornstein-Uhlenbeck process) which is required for exploration
class OU_Noise(object):

    # Defining of the variables according to the Ornstein-Unlenbeck process and required values selected are
    # accordingly defined in the paper
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=0.001, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    # Function call to return the updated x
    def __call__(self):
        x = self.prev_x + self.theta*(self.mu - self.prev_x)*self.dt + \
            self.sigma*np.sqrt(self.dt)*np.random.normal(size=self.mu.shape)
        self.prev_x = x
        return x

    # Reset the parameter to the initial conditions or defining zeros if no value is present
    def reset(self):
        if self.x0 is not None:
            self.prev_x = self.x0
        else:
            self.prev_x = np.zeros_like(self.mu)

# Work in Progress

class Sample_Buffer(object):

class Replay_Buffer(object):

class Actor_Network(object):

class Critic_Network(object):
    print ("Critic")