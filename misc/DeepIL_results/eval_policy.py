import argparse
import torch

from ai_coach_core.model_learning.DeepIL.env import make_env
from ai_coach_core.model_learning.DeepIL.algo.algo import EXP_ALGOS
from ai_coach_core.model_learning.DeepIL.utils import (evaluation,
                                                       state_action_size)
import latent_config as lc


def run(args):
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
    algo.set_reward(lat_conf.get_reward)
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

  mean_return = evaluation(env=env,
                           algo=algo,
                           episodes=args.episodes,
                           render=args.render,
                           seed=args.seed,
                           delay=args.delay)

  print(mean_return)


if __name__ == '__main__':
  p = argparse.ArgumentParser()

  # required
  p.add_argument('--actor-weights',
                 type=str,
                 required=True,
                 help='path to the well-trained actor weights of the agent')
  p.add_argument('--env-id',
                 type=str,
                 required=True,
                 help='name of the environment')
  p.add_argument('--algo',
                 type=str,
                 required=True,
                 help='name of the well-trained agent')

  # custom
  p.add_argument('--render',
                 action='store_true',
                 default=False,
                 help='render the environment or not')
  p.add_argument('--trans-weights',
                 type=str,
                 default="",
                 help='path to the well-trained trans weights of the agent')

  # default
  p.add_argument('--episodes',
                 type=int,
                 default=10,
                 help='number of episodes used in evaluation')
  p.add_argument('--seed', type=int, default=0, help='random seed')
  p.add_argument('--delay',
                 type=float,
                 default=0,
                 help='number of seconds to delay while rendering,'
                 'in case the agent moves too fast')

  args = p.parse_args()
  run(args)
