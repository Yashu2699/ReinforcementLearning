import gym
import numpy as np
from agent_td3 import Agent
from utils import plot_learning_curve

if __name__ == '__main__':

	env = gym.make('BipedalWalker-v3')
	agent = Agent(alpha=0.001, beta=0.001, input_dims=env.observation_space.shape,
				tau=0.005, env=env, batch_size=100, fc1_dims=400, fc2_dims=300,
				n_actions=env.action_space.shape[0])

	num_games = 1500
	filename = 'BipedalWalker.png'
	figure_file = 'plots/' + filename

	best_score = env.reward_range[0]
	score_history = []

	for i in range(num_games):
		observation = env.reset()
		done = False
		score = 0
		while not done:
			action = agent.choose_action(observation)
			observation_, reward, done, info = env.step(action)
			agent.remember(observation, action, reward, observation_, done)
			agent.learn()
			score += reward
			observation = observation_
		score_history.append(score)
		avg_score = np.mean(score_history[-100:])

		if avg_score > best_score:
			best_score = avg_score
			agent.save_models()

		print('episode ', i, 'score %.2f' % score, 'average %.3f' % avg_score)

	x = [i+1 for i in range(num_games)]
	plot_learning_curve(x, score_history, figure_file)