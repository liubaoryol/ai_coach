import gym
import gym_custom
import numpy as np


def action_fn(obs, env):
  action_space = env.action_space
  action_space_mean = (action_space.low + action_space.high) / 2.0
  action_space_magn = (action_space.high - action_space.low) / 2.0
  random_action = (
      action_space_mean + action_space_magn *
      np.random.uniform(low=-1.0, high=1.0, size=action_space.shape))
  return random_action


if __name__ == "__main__":
  import time

  num_episodes = 1
  episode_length = 1000

  env = gym.make('AntPush-v0')

  rewards = []
  for ep in range(num_episodes):
    rewards.append(0.0)
    obs = env.reset()
    for _ in range(episode_length):
      env.render()
      time.sleep(2)
      obs, reward, done, _ = env.step(action_fn(obs, env))
      rewards[-1] += reward
      if done:
        break
