
import numpy as np
import random
import copy
from collections import namedtuple, deque

import importlib
import model
importlib.reload(model)
from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
LR_ACTOR = 1e-3         # learning rate of the actor
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
STEPS_TO_UPDATE = 1         # learning timestep interval
UPDATE_TIMES = 1           # number of learning passes
GAMMA = 0.99            # discount factor
TAU = 8e-3              # for soft update of target parameters
OU_SIGMA = 0.2          # Ornstein-Uhlenbeck noise parameter, volatility
OU_THETA = 0.15         # Ornstein-Uhlenbeck noise parameter, speed of mean reversion
EPS_START = 1.0           # explore->exploit noise process added to act step
EPSILON_DECAY = 1.0 # 0.999

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agents():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, num_agents, random_seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        
        
        ''' 
        TODO: 
        1. Set up 'N' agents, each with their own local, target, optimizers.
        2. All agents use the same actor network, but it is conditioned on the state of all actors.
        3. All agents use a shared critic network, but the critic network estimates the collective 
        value of all states of all agents.
        
        '''
        
        # Set up the shared hyperparameters among all agents.
        self.steps = 0
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.num_agents = num_agents
        self.eps = EPS_START
        self.eps_decay = EPSILON_DECAY
        
        # Set up the actor and critic networks
        # Actor Network (w/ Target Network)
        self.actor_local_list = []
        self.actor_target_list = []
        self.actor_optimizer_list = []
        for i in range(num_agents):
            self.actor_local_list.append(Actor(state_size, action_size, num_agents, random_seed).to(device))
            self.actor_target_list.append(Actor(state_size, action_size, num_agents, random_seed).to(device))
            self.actor_optimizer_list.append(optim.Adam(self.actor_local_list[i].parameters(), lr=LR_ACTOR))

        # Make a critic per agent.
        self.critic_local_list = []
        self.critic_target_list = []
        self.critic_optimizer_list = []
        for i in range(num_agents):
            # Critic Network (w/ Target Network)
            self.critic_local_list.append(Critic(state_size, action_size,  num_agents, random_seed).to(device))
            self.critic_target_list.append(Critic(state_size, action_size, num_agents, random_seed).to(device))
            self.critic_optimizer_list.append(optim.Adam(self.critic_local_list[i].parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY))

        # Separate noise process for each agent so that their learning isn't correlated, 
        # which will hopefully lead to more learning and exploration.
        self.noise_list = []
        for i in range(num_agents):
            self.noise_list.append(OUNoise(action_size, random_seed))

        # Replay memory for each agent.
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
    
    def step(self, state_set, action_set, reward_set, next_state_set, done):
        
        ''' 
        TODO: 
        1. Each tuple added to the memory will now be (state, state set w/state zero'd , action, reward, next_state, done)
        '''
        
        
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state_set, action_set, reward_set, next_state_set, done)

        # Increment step counter.
        self.steps += 1
        
        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE and self.steps % STEPS_TO_UPDATE == 0:
            for i in range(UPDATE_TIMES):
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, states, add_noise=True):
        """Returns actions for given states for each agent as per current policy."""
        actions = []
        for i in range(self.num_agents):
            state = torch.from_numpy(states).float().to(device)
            self.actor_local_list[i].eval()
            with torch.no_grad():
                action = self.actor_local_list[i](state).cpu().data.numpy()
            self.actor_local_list[i].train()
            if add_noise:
                noise = self.noise_list[i].sample()
                action += self.eps * noise
            actions.append(action)
        actions = np.array(actions).reshape((self.num_agents, self.action_size))
        return np.clip(actions, -1, 1)

    def reset(self):
        for i in range(self.num_agents):
            self.noise_list[i].reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        
        ''' 
        TODO: 
        1. Set up 'N' agents, each with their own local, target, optimizers.
        2. All agents use the same actor network, but it is conditioned on the state of all actors.
        3. All agents use a shared critic network, but the critic network estimates the collective 
        value of all states of all agents.
        
        Update the learning estimates so that the agents 
        
        '''
   
        state_sets, action_sets, reward_sets, next_state_sets, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        
        # GENERATE NEXT ACTIONS FOR EACH AGENT. a'1, a'2, ..., a'N
        # Convert the matrix into a list of individual states to predict actions for.
        next_states = next_state_sets.reshape((BATCH_SIZE*self.num_agents, self.state_size))
        # Predict actions for each state in the list.
        actions_next_list = []
        # self.num_agents
        for i in range(self.num_agents):
            # Get the next action for each next state set for agent i.
            actions_next = self.actor_target_list[i](next_state_sets)
            # Add the next actions for this agent to the list.
            actions_next_list.append(actions_next.reshape(-1, self.action_size))
            
        # Roll the actions into next_action_sets by using the torch cat command.
        action_next_sets = torch.cat(tuple(actions_next_list), dim = 1)
            
        # FOR EACH AGENT, UPDATE THE AGENT'S CORRESPONDING CRITIC.
        for i in range(self.num_agents):           
            Q_targets_next = self.critic_target_list[i](next_state_sets, action_next_sets)
            # Compute Q targets for current states (y_i)      
            Q_targets = reward_sets[:, i].reshape((-1, 1)) + (gamma * Q_targets_next * (1 - dones[:, i].reshape((-1,1))))
            # Compute critic loss
            Q_expected = self.critic_local_list[i](state_sets, action_sets)
            critic_loss = F.mse_loss(Q_expected, Q_targets)
            # Minimize the loss
            self.critic_optimizer_list[i].zero_grad()
            critic_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.critic_local_list[i].parameters(), 1)
            self.critic_optimizer_list[i].step()

            # ---------------------------- update actor ---------------------------- #
            # Select only states related to agent i.
            states = state_sets.reshape((BATCH_SIZE*self.num_agents, self.state_size))
            inx = np.array([k for k in range(len(states)) if k%(self.num_agents)==i])
            
            # Compute action predictions for current agent and its states.
            actions_pred = self.actor_local_list[i](state_sets)
            
            # Copy action_sets, and reset the agent's actual actions with its predicted actions.
            actions = action_sets.clone().reshape((BATCH_SIZE*self.num_agents, self.action_size))
            actions[inx] = actions_pred
            
            # Roll the actions back up into the expected action_sets format.
            actions_pred_sets = actions.reshape((BATCH_SIZE, self.action_size * self.num_agents))          
            
            # Compute the actor loss given the state sets and predicted action sets.
            actor_loss = -self.critic_local_list[i](state_sets, actions_pred_sets).mean()
            # Minimize the loss
            self.actor_optimizer_list[i].zero_grad()
            actor_loss.backward(retain_graph=True)
            self.actor_optimizer_list[i].step()

        for i in range(self.num_agents):
            # ----------------------- update target networks ----------------------- #
            self.soft_update(self.critic_local_list[i], self.critic_target_list[i], TAU)
            # NOTE: Possibly move this out so the actor network only gets updated once.
            self.soft_update(self.actor_local_list[i], self.actor_target_list[i], TAU)

        # Decay noise and reset.
        self.eps = self.eps * EPSILON_DECAY
        self.reset()

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=OU_THETA, sigma=OU_SIGMA):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state_set", "action_set", "reward_set", "next_state_set", "dones"])
        self.seed = random.seed(seed)
    
    def add(self, state_set, action_set, reward_set, next_state_set, done):
        """Add a new experience to memory."""
        e = self.experience(state_set, action_set, reward_set, next_state_set, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        state_sets = torch.from_numpy(np.vstack([e.state_set for e in experiences if e is not None])).float().to(device)
        action_sets = torch.from_numpy(np.vstack([e.action_set for e in experiences if e is not None])).float().to(device)
        reward_sets = torch.from_numpy(np.vstack([e.reward_set for e in experiences if e is not None])).float().to(device)
        next_state_sets = torch.from_numpy(np.vstack([e.next_state_set for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.dones for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (state_sets, action_sets, reward_sets, next_state_sets, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


