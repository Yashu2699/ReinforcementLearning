import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class CriticNetwork(nn.Module):
	def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions,
				name, chkpt_dir='tmp/td3'):
		super(CriticNetwork, self).__init__()

		self.input_dims = input_dims
		self.fc1_dims = fc1_dims
		self.fc2_dims = fc2_dims
		self.n_actions = n_actions
		self.chkpt_dir = chkpt_dir
		self.checkpoint_file = os.path.join(self.chkpt_dir, name+'_td3')

		self.fc1 = nn.Linear(self.input_dims[0] + n_actions, self.fc1_dims)
		self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
		self.q = nn.Linear(self.fc2_dims, 1)

		self.optimizer = optim.Adam(self.parameters(), lr=beta)
		self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
		self.to(self.device)

	def forward(self, state, action):
		q1 = self.fc1(T.cat([state, action], dim=1))
		q1 = F.relu(q1)
		q1 = F.relu(self.fc2(q1))
		q = self.q(q1)
		return q

	def save_checkpoint(self):
		print('.. Saving Checkpoint ..')
		T.save(self.state_dict(), self.checkpoint_file)

	def load_checkpoint(self):
		print('.. Loading Checkpoint ..')
		self.load_state_dict(T.load(self.checkpoint_file))

class ActorNetwork(nn.Module):
	def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions,
				name, chkpt_dir='tmp/td3'):
		super(ActorNetwork, self).__init__()

		self.input_dims = input_dims
		self.fc1_dims = fc1_dims
		self.fc2_dims = fc2_dims
		self.n_actions = n_actions
		self.name = name
		self.chkpt_dir = chkpt_dir
		self.checkpoint_file = os.path.join(self.chkpt_dir, name+'_td3')

		self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
		self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
		self.mu = nn.Linear(self.fc2_dims, self.n_actions)

		self.optimizer = optim.Adam(self.parameters(), lr=alpha)
		self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
		self.to(self.device)

	def forward(self, state):
		prob = F.relu(self.fc1(state))
		prob = F.relu(self.fc2(prob))
		prob = T.tanh(self.mu(prob))

		return prob

	def save_checkpoint(self):
		print('.. Saving Checkpoint ..')
		T.save(self.state_dict(), self.checkpoint_file)

	def load_checkpoint(self):
		print('.. Loading Checkpoint ..')
		self.load_state_dict(T.load(self.checkpoint_file))