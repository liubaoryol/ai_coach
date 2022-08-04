import argparse
import torch
import cv2
import numpy as np
import time

from ai_coach_core.model_learning.DeepIL.env import make_env
import latent_config as lc


def run(args):
  """Collect demonstrations"""
  env = make_env(args.env_id)

  lat_conf = lc.LATENT_CONFIG[args.env_id]  # type: lc.LatentConfig

  env.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  torch.cuda.manual_seed(args.seed)

  state = env.reset()
  t = 0
  latent = lat_conf.get_latent(0, state, None, None, None)
  episode_return = 0.0
  episode_steps = 0

  env.render()
  while True:
    key = cv2.waitKey(100)
    action = None
    if key == 83:  # right
      action = np.array([1.0, 0.0])
    elif key == 81:  # left
      action = np.array([-1.0, 0.0])
    elif key == 82:  # up
      action = np.array([0.0, -1.0])
    elif key == 84:  # down
      action = np.array([0.0, 1.0])
    elif key == 27:  # esc
      break

    if action is not None:
      t += 1
      next_state, reward, done, _ = env.step(action)
      reward = lat_conf.get_reward(state, latent, action, reward)
      print(f'State: {state}, Latent: {latent}, '
            f'Action: {action}, Next: {next_state}')
      print(f'Instant reward: {reward}')

      next_latent = lat_conf.get_latent(t, next_state, latent, action, state)
      episode_return += reward
      episode_steps += 1
      state = next_state
      latent = next_latent

      # Render the game
      env.render()
      time.sleep(0.3)

      if done or t == env.max_episode_steps:
        state = env.reset()
        t = 0
        latent = lat_conf.get_latent(0, state, None, None, None)
        print(f'Episode steps: {episode_steps}')
        print(f'Episode return: {episode_return}')
        episode_return = 0.0
        episode_steps = 0
        break


if __name__ == '__main__':
  p = argparse.ArgumentParser()

  # required
  p.add_argument('--env-id',
                 type=str,
                 required=True,
                 help='name of the environment')

  # default
  p.add_argument('--seed', type=int, default=0, help='random seed')

  args = p.parse_args()
  run(args)
