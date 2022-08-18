import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Network(nn.Module):
	def __init__(self, lr, input_dims, n_actions, fc1_dims=256, fc2_dims=256):
		super(Network, self).__init__()
		self.fc1 = nn.Linear(*input_dims, fc1_dims)
		self.fc2 = nn.Linear(fc1_dims, fc2_dims)
		self.pi = nn.Linear(fc2_dims, n_actions)
		self.v = nn.Linear(fc2_dims, 1)
		self.optimizer = optim.Adam(self.parameters(), lr=lr)
		self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
		self.to(self.device)

	def forward(self, state):
		fc1 = F.relu(self.fc1(state))
		fc2 = F.relu(self.fc2(fc1))
		pi = self.pi(fc2)
		v = self.v(fc2)

		return (pi, v)

# no memory before this is a single step td method

class Agent():
	def  __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions, gamma=0.99):
		self.gamma = gamma
		self.lr = lr
		self.fc1_dims = fc1_dims
		self.fc2_dims = fc2_dims

		self.actor_critic = Network(lr, input_dims, n_actions, self.fc1_dims,
									self.fc2_dims)

		self.log_prob = None # do not need to store a list

	def choose_action(self, observation):
		state = T.tensor([observation], dtype=T.float).to(self.actor_critic.device)
		probabilities, _ = self.actor_critic.forward(state)
		probabilities = F.softmax(probabilities, dim=1)
		action_probs = T.distributions.Categorical(probabilities)
		action = action_probs.sample()
		log_probs = action_probs.log_prob(action)
		self.log_prob = log_probs

		return action.item()

	def learn(self, state, reward, state_, done):
		self.actor_critic.optimizer.zero_grad()

		state = T.tensor([state], dtype=T.float).to(self.actor_critic.device)
		state_ = T.tensor([state_], dtype=T.float).to(self.actor_critic.device)
		reward = T.tensor([reward], dtype=T.float).to(self.actor_critic.device)

		_, critic = self.actor_critic.forward(state)
		_, critic_ = self.actor_critic.forward(state_)

		delta = reward + self.gamma * critic_*(1 - int(done)) - critic

		actor_loss = -self.log_prob * delta
		critic_loss = delta**2

		(actor_loss + critic_loss).backward()
		self.actor_critic.optimizer.step()
