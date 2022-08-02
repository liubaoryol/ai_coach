import os
import argparse
import torch

from ai_coach_core.model_learning.DeepIL.env import make_env
from ai_coach_core.model_learning.DeepIL.algo.algo import EXP_ALGOS
from ai_coach_core.model_learning.DeepIL.utils import (collect_demo,
                                                       state_action_size)
import latent_config as lc


def run(args):
  """Collect demonstrations"""
  env = make_env(args.env_id)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  (state_size, discrete_state, action_size,
   discrete_action) = state_action_size(env)

  lat_conf = lc.LATENT_CONFIG[args.env_id]  # type: lc.LatentConfig

  if args.algo == "ppo":
    algo = EXP_ALGOS[args.algo](state_size=state_size,
                                latent_size=lat_conf.latent_size,
                                action_size=action_size,
                                discrete_state=discrete_state,
                                discrete_latent=lat_conf.discrete_latent,
                                discrete_action=discrete_action,
                                cb_get_latent=lat_conf.get_latent,
                                device=device,
                                path_actor=args.actor_weights,
                                units_actor=(64, 64))
  elif args.algo in ["digail", "vae"]:
    algo = EXP_ALGOS[args.algo](state_size=state_size,
                                latent_size=lat_conf.latent_size,
                                action_size=action_size,
                                discrete_state=discrete_state,
                                discrete_latent=lat_conf.discrete_latent,
                                discrete_action=discrete_action,
                                cb_init_latent=lat_conf.get_init_latent,
                                device=device,
                                path_actor=args.actor_weights,
                                path_trans=args.trans_weights,
                                units_actor=(64, 64),
                                units_trans=(64, 64))

  buffer, mean_return = collect_demo(env=env,
                                     latent_size=lat_conf.latent_size,
                                     discrete_latent=lat_conf.discrete_latent,
                                     algo=algo,
                                     buffer_size=args.buffer_size,
                                     device=device,
                                     p_rand=args.p_rand,
                                     seed=args.seed)

  if os.path.exists(
      os.path.join(
          'buffers/Raw', args.env_id,
          f'size{args.buffer_size}_reward{round(mean_return, 2)}.pth')):
    print('Error: demonstrations with the same reward exists')
  else:
    buffer.save(
        os.path.join(
            'buffers/Raw', args.env_id,
            f'size{args.buffer_size}_reward{round(mean_return, 2)}.pth'))


if __name__ == '__main__':
  p = argparse.ArgumentParser()

  # required
  p.add_argument('--actor-weights',
                 type=str,
                 required=True,
                 help='path to the well-trained actor weights of the agent')
  p.add_argument('--trans-weights',
                 type=str,
                 required=True,
                 help='path to the well-trained trans weights of the agent')
  p.add_argument('--env-id',
                 type=str,
                 required=True,
                 help='name of the environment')
  p.add_argument('--algo',
                 type=str,
                 required=True,
                 help='name of the well-trained agent')

  # default
  p.add_argument('--buffer-size',
                 type=int,
                 default=40000,
                 help='size of the buffer')
  p.add_argument(
      '--p-rand',
      type=float,
      default=0.0,
      help='with probability of p_rand, the policy will act randomly')
  p.add_argument('--seed', type=int, default=0, help='random seed')

  args = p.parse_args()
  run(args)
