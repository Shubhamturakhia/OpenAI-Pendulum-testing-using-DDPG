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