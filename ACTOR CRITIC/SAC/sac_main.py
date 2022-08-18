import pybullet_envs
print('Here')
import gym
import numpy as np
from sac_agent import Agent
from utils import plot_learning_curve

if __name__ == '__main__':

	#env_id = 'InvertedPendulumBulletEnv-v0'
	env_id = 'LunarLanderContinuous-v2'
	env = gym.make(env_id)
	agent = Agent(alpha=0.0003, beta=0.0003, reward_scale=2, env_id=env_id,
				input_dims=env.observation_space.shape, tau=0.005, env=env,
				batch_size=256, fc1_dims=256, fc2_dims=256,
				n_actions=env.action_space.shape[0])

	num_games = 250
	file_name = 'sac.png'
	figure_file = 'plots/' + file_name

	best_score = env.reward_range[0]
	score_history = []
	
	steps = 0

	for i in range(num_games):
		observation = env.reset()
		done = False
		score = 0
		while not done:
			action = agent.choose_action(observation)
			observation_, reward, done, info = env.step(action)
			steps += 1
			agent.remember(observation, action, reward, observation_, done)
			agent.learn()
			score += reward
			observation = observation_
		score_history.append(score)
		avg_score = np.mean(score_history[-100:])

		if avg_score > best_score:
			best_score = avg_score
			agent.save_models()

		print('episode ', i, 'score %.2f' %score, 'avg score %.2f' %avg_score,
			'best_score %.2f' % best_score)

	x = [i+1 for i in range(num_games)]
	plot_learning_curve(x, score_history, figure_file)