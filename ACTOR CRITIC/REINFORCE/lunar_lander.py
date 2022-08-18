import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class PolicyNetwork(nn.Module):
	def __init__(self, lr, input_dims, n_actions):
		super(PolicyNetwork, self).__init__()
		self.fc1 = nn.Linear(*input_dims, 128)
		self.fc2 = nn.Linear(128, 128)
		self.fc3 = nn.Linear(128, n_actions)

		self.optimizer = optim.Adam(self.parameters(), lr=lr)
		self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
		self.to(self.device)

	def forward(self, state):
		fc1 = F.relu(self.fc1(state))
		fc2 = F.relu(self.fc2(fc1))
		fc3 = self.fc3(fc2)

		return fc3

class Agent():
	def __init__(self, lr, input_dims, gamma=0.99, n_actions=4):
		self.gamma = gamma
		self.lr = lr
		self.reward_memory = []
		self.action_memory = [] # stores log probs of actions the agent took

		self.policy = PolicyNetwork(self.lr, input_dims, n_actions)

	def choose_action(self, observation):
		state = T.tensor([observation]).to(self.policy.device)
		probabilities = F.softmax(self.policy.forward(state))
		action_probs = T.distributions.Categorical(probabilities)
		action = action_probs.sample()
		log_probs = action_probs.log_prob(action)
		self.action_memory.append(log_probs)

		return action.item()

	def store_rewards(self, reward):
		self.reward_memory.append(reward)

	def learn(self):
		self.policy.optimizer.zero_grad()

		G = np.zeros_like(self.reward_memory)
		for t in range(len(self.reward_memory)):
			G_sum = 0
			discount = 1
			for k in range(t, len(self.reward_memory)):
				G_sum += self.reward_memory[k] * discount
				discount *= self.gamma
			G[t] = G_sum
		G = T.tensor(G, dtype=T.float).to(self.policy.device)

		loss = 0
		for g, logprob in zip(G, self.action_memory):
			loss += -g * logprob

		loss.backward()
		self.policy.optimizer.step()

		self.action_memory = []
		self.reward_memory = []