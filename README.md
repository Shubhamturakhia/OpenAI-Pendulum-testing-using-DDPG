## Reinforcement-Learning-for-Bike-Control
Masters Thesis Project: Fall 2020

Thesis Title: Bike Control and Navigation using Reinforcement Learning

Paper referred for the DDPG algorithm: "**https://arxiv.org/pdf/1509.02pdf971.**"

### Algorithm Design Procedure: 
Following the DDPG algorithm procedure, we need the old state, action, reward and new states as
output for every step

##### Things required for Algorithm processing:
1. Class for Storing the previous values (rewards and states) -
2. Class for Actor DNN : class Actor(Obj)
3. Class for Critic DNN: class Critic(Obj)
4. Class for Ornstein Unlenbeck (this would be a class defined for Noise) - Ornstein-Uhlenbeck process (Uhlenbeck &
Ornstein, 1930) - page 4 - (https://arxiv.org/pdf/1509.02pdf971.)
5. Class or function for Memory size and/or terminal size
6. Loading the environment and deployment of Algorithm.
7. Class for Agent to do the learning  and make use of the classes above: class Agent(Obj)

##### Constraints required:
1. Deterministic policy is action based (i.e it O/P is "Action" and Not a "Prob value")
2. Limiting the constraints in the environment designed

##### Steps for running the code and features

