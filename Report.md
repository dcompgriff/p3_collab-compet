# P3 - Collaboration and Competition Project

Author: Daniel Griffin

## Problem Overview

The general goal of this project is to develop a multi-agent reinforcement learning algorithm for a collaborative task. How to best train agents in a multi-agent environment is still an open problem, and has many challenges and variations. Most environments that agent's need to learn to act in have multiple agents with a combination of collaborative and adversarial objectives. Learning to act in these advanced environments is a major step towards general AI systems. 

In this project, the goal was to develop a multi-agent reinforcement learning algorthm that could learn to collaboratively play tenis with another agent. The goal of the agents was to keep passing the tennis ball between both agents for as long as possible without the ball exiting the court or dropping on the court. I developed a slight variant of the MADDPG algorithm described in [this](https://arxiv.org/pdf/1706.02275.pdf) paper to solve this problem. The details of the environment, algorithmic approach, results, and future work are described below. 

## Learning Environment

The environment is similar to the Unity ML Agents 'Tennis' environment outlined [here](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis). This environment contains two agents that control rackets. Their collective objective is to collaborate together to keep the ball bouncing over the net for as long as possible. If an agent hits a ball over the net, it recieves a reward of +0.1, and if the agent lets the ball hit the ground, it recieves a reward of -0.1. Thus, the goal of an agent is to explicitly hit the ball over the net as many times as possible. The best strategy to do this is implicitly defined in the reward structure. If the agent's can work together to keep hitting the ball back and forth to each other, then they can collectively maximize their individual rewards. For the environment to be solved, the agents must get an average score of +0.5 over 100 consecutive episodes where the score for each individual episode is taken as the maximum of the scores returned from both agents. 

States: 24 dimension continuous vector
Actions: 2 dimension continuous vectors between [-1, +1]
Rewards: -0.1 when the ball touches the ground on the agent's side, +0.1 when the ball is hit over the net, 0.0 otherwise.

## Algorithm

### DDPG

To solve this problem, I modified my original implementation of DDPG [here](https://github.com/dcompgriff/p2_continuous_control) to use multiple agents to optimize the environment. For a full description of the algorithm and it's implementation in the single agent case, see [this](https://github.com/dcompgriff/p2_continuous_control/blob/master/Report.md) report.

### MADDPG

To solve the overall problem, I implemented a variant of the MADDPG algorithm. The major difference was that my implementation allowed the actor policy for each agent to see the current state of all agents in the environment. In the original paper, the authors assumed that agents only had access to the state of other agents at training time for use with the critic, and not after. The main reason for this variation was to explicitly guide the actor policies with more context information. While it is possible that the actors could implicitly attempt to learn how to act given only their own view of the environment, it takes more data to do so. The key challenge here was that if both agent's didn't have a successful hit of the ball over the net, one agent would learn to dominate the other, with the weaker agent never learning any actions that have value. The original algorithm structure, along with my variation is given below.

![alt text](https://github.com/dcompgriff/p3_collab-compet/blob/master/MADDPG_RL_Structure.png "MADDPG Original Algorithm")

![alt text](https://github.com/dcompgriff/p3_collab-compet/blob/master/MY_MADDPG_RL_Structure.png "MADDPG Variant Algorithm")


### Approaches

1. I initially tried using a single critic function without visibility to other states and actions, and a single actor with visibility into other states.
    * This Failed! It makes sense why it would fail because the best total reward we wanted was due to collaboration, and it's hard for agents to find policies that explicitly learn this collaborative nature. It's possible to do so, but unlikely given the greedy nature of the policies.
    
2. I then tried using a single actor with multiple critics, each with visibility into the agent's states and next actions. 
    * This Failed! The agent's actions were so similar that they often ended up mirroring each other, leading to poor performance, as agent's typically need to mirror each other along one dimension, while inversely mirroring along another. 
 
3. Next, I tried a single actor with multiple critics, each with visibility into the agent's states and next actions. 
    * This Failed! This is the full MADDPG implementation (as actors only use their local observations). It still failed becaues it's still too hard to collaborate when the agents actor policies only have implicit information into the correct action to take, and have no visibility into other actor's states to condition on.
 
4. In the final implementation, I created multiple agents with multiple critics, and gave both full visibility into the other agent's states and actions.
    * This was Successful! The agents and critics needed to have full visibility into the state of both agents, along with a critic that evaluated both agents. This was also coupled with the need for each agent learning their own, unique policy, and to explore the action space with their own individual noise processes.

### Key components to success

1. The right level of visibility of the actors and critics in the approaches was critical. Both actors and critics needed to see all agent states and actions.

2. Having an actor and critic per agent was also critical for tuning an individual agent's performance. Sharing actors and critics didn't work well. 

3. The right noise level was also critical. As in p2, the right noise level was critical to successfully exploring the action space. The original parameters for the noise process were way too high, causing saturated action values or fast actions which prevented the agents from being able to hit the initial ball over the net.

4. Getting an initial set of 300-ish episodes where both agents win a few times was also critical. One agent always wins first and then tends to dominate, and if the exploration isn't tuned correctly, it may always dominate, with the other agent never winning and never learning from those winning experiences. Once both sets of agents have won a few times (Aka gotten a value of >0.0 as their total cumulative reward), they can cooperate on their experience to keep feeding each other better and better collective rewards to learn from. 

### Final Model and Hyperparameters

#### Actor Model

The actor model was a standard neural network model that accepted a concatenated vector of all agent states as input, and output the values of the best action to take. (FCxN = Fully Connected Layer with N neurons)

* num layers: 3
* layer 1: Input (48x1 states) -> FCx128 -> ReLU -> Batch Norm
* layer 2: FCx64 -> ReLU
* layer 3: FCx2 -> Tanh


#### Critic Model

The critic model was a standard neural network model that accepted a concatenated vector of all agent states as input, along with an action vector. The network outputs the 'Q-Value' of the best action, or the estimate of the best action to take. (FCxN = Fully Connected Layer with N neurons)

* num layers: 3
* layer 1: Input (48x1 states) -> FCx128 -> ReLU -> Batch Norm
* layer 2: Input [(128x1 layer1 output), (2x1 action vector)] -> FCx64 -> ReLU
* layer 3: FCx1

#### DDPG Algorithm Hyperparameters

* BUFFER_SIZE = int(1e6)  # replay buffer size
* BATCH_SIZE = 128        # minibatch size
* LR_ACTOR = 1e-3         # learning rate of the actor
* LR_CRITIC = 1e-3        # learning rate of the critic
* WEIGHT_DECAY = 0        # L2 weight decay
* STEPS_TO_UPDATE = 1         # learning timestep interval
* UPDATE_TIMES = 1           # number of learning passes
* GAMMA = 0.99            # discount factor
* TAU = 8e-3              # for soft update of target parameters
* OU_SIGMA = 0.2          # Ornstein-Uhlenbeck noise parameter, volatility
* OU_THETA = 0.15         # Ornstein-Uhlenbeck noise parameter, speed of mean reversion
* EPS_START = 1.0           # explore->exploit noise process added to act step
* EPSILON_DECAY = 1.0 # exponential decay process
* Max Episode Length = 5000 # Number of steps to run the environment before closing. 

## Final Results

The agent achieved a performance of > 0.5 on average after 798 episodes. This performance is shown in the first image below. To test the full performance, I let the agent run for 1500 training episodes. The agents saturated performance at a max score of 2.6, the performance for which is shown in the second image below.

![alt text](https://github.com/dcompgriff/p3_collab-compet/blob/master/full_performance_chart.png "MADDPG Variant Algorithm Performance")

![alt text](https://github.com/dcompgriff/p3_collab-compet/blob/master/long_term_performance_chart.png "MADDPG Variant Algorithm Saturated Performance")


## Ideas for Future Work

* Better noise process.
    * The noise process and exploration process for continuous actions in general is pretty terrible for general learning. A better, more systematic method should be used.
* Skewed Training data set.
    * Training methods that deal with skewed training data sets (over sampling and under sampling) should be used. This skew in performance is a huge impediment for the system's actor and critic models to learn.
* Prioritized replay buffer.
    * Along the same vane as the skewed training data set issue, using a prioritized reply buffer would be a huge help for improving the rate of learning.
* Transfer learning of actor and critic mode from one agent to another.
    * Possibly prioritize or weight the experiences in one agent by similarity to experiences in another. Possibly even use a multi-sampling memory where experiences are sampled across agent memories.
* Transfer learning for action selection. 
    * Adjust the actor probabalistic policy to prioitize actions that worked for another agent when in a similar state.





























MARL



























