import gym
from blackjack_prediction import Agent 

if __name__ == '__main__':
	env = gym.make('Blackjack-v1')
	agent = Agent()
	n_episodes = 500_000

	for i in range(n_episodes):
		if i % 5000 == 0:
			print('starting episode ', i)
		observation = env.reset()
		done = False

		while not done:
			action = agent.policy(observation)
			observation_, reward, done, info = env.step(action)
			agent.memory.append((observation, reward))
			observation = observation_
		agent.update_V()

	print(agent.V[(21, 3, True)])
	print(agent.V[(4, 1, False)])