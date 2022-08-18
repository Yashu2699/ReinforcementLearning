import numpy as np

class Agent():
	def __init__(self, gamma=0.99):

		self.V = {} # Agent's estimate of the value of states
		self.sum_space = [i for i in range(4, 22)]
		self.dealer_show_card = [i+1 for i in range(10)]
		self.ace_space = [False, True]
		self.action_space = [0, 1] # stick or hit

		self.state_space = []
		self.returns = {}
		self.states_visited = {}
		self.memory = []
		self.gamma = gamma

		self.init_vals()

	def init_vals(self):
		for total in self.sum_space:
			for card in self.dealer_show_card:
				for ace in self.ace_space:
					self.V[(total, card, ace)] = 0
					self.returns[(total, card, ace)] = []
					self.states_visited[(total, card, ace)] = 0
					self.state_space.append((total, card, ace))

	def policy(self, state):
		total, _, _ = state
		action = 0 if total >= 20 else 1
		return action

	def update_V(self):
		for idt, (state, _) in enumerate(self.memory):
			G = 0
			if self.states_visited[state] == 0:
				self.states_visited[state] += 1
				discount = 1
				for t, (_, reward) in enumerate(self.memory[idt:]):
					G += reward * discount
					discount *= self.gamma
					self.returns[state].append(G)

		for state, _ in self.memory:
			self.V[state] = np.mean(self.returns[state])


		# Resets after an episode is over
		for state in self.state_space:
			self.states_visited[state] = 0

		self.memory = []

# import gym

# if __name__ == '__main__':
# 	env = gym.make('Blackjack-v0')
# 	agent = Agent()
# 	n_episodes = 500_000

# 	for i in range(n_episodes):
# 		if i % 5000 == 0:
# 			print('starting episode ', i)
# 		observation = env.reset()
# 		done = False

# 		while not done:
# 			action = agent.policy(observation)
# 			observation_, reward, done, info = env.step(action)
# 			agent.memory.append((observation, reward))
# 			observation = observation_
# 		agent.update_V()

# 	print(agent.V[(21, 3, True)])
# 	print(agent.V[(4, 1, False)])