# mario has 256 discrete actions 
# when we wrap it around joypadspace and pass in simple movement param, 
#the action space is discrete(7)


!pip install gym_super_mario_bros==7.3.0 nes_py

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

done = True
for step in range(100_000):
	if done:
		env.reset()
	state, reward, done, info = env.step(env.action_space.sample())
	env.render()

env.close()


# Preprocess Env

!pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio===0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
!pip install stable-baselines3[extra]

from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from matplotlib import pyplot as plt

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = GrayScaleObservation(env, keep_dim=True)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, 4, channels_order='last')

# Training model

import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True



CHECKPOINT_DIR = './train/'
LOG_DIR = './logs/'

callback = TrainAndLoggingCallback(check_freq=100000, save_path=CHECKPOINT_DIR)

model = PPO('CnnPolicy', env, verobse=1, tensorboard_log=LOG_DIR,
			learning_rate=0.000001, n_steps=512)

model.learn(total_timesteps=1000000, callback=callback)


# Testing model

model = PPO.load('./train/best_model_1000000')
state = env.reset()

while true:
	action, _ = model.predict(state)
	new_state, reward, done, info = env.step(action)
	state = new_state
	env.render()
env.close()