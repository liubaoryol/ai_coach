import os
import argparse
import torch
from torch import nn
import numpy as np

from datetime import datetime
from ai_coach_core.model_learning.DeepIL.env import make_env
from ai_coach_core.model_learning.DeepIL.utils import state_action_size
from ai_coach_core.model_learning.DeepIL.buffer import (SerializedBuffer,
                                                        RolloutBuffer)
from ai_coach_core.model_learning.DeepIL.algo.algo import ALGOS
from ai_coach_core.model_learning.DeepIL.trainer import Trainer
from ai_coach_core.model_learning.DeepIL.network import (
    GAILDiscrim, DiscreteTransition, DiscretePolicy, StateFunction,
    ContinousPolicy, ContinousTransition)
import latent_config as lc


def run(args):
  """Train Imitation Learning algorithms"""
  env = make_env(args.env_id)
  env_test = env
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  torch.manual_seed(args.seed)
  np.random.seed(args.seed)

  (state_size, discrete_state, action_size,
   discrete_action) = state_action_size(env)

  lat_conf = lc.LATENT_CONFIG[args.env_id]  # type: lc.LatentConfig

  buffer_exp = SerializedBuffer(path=args.buffer,
                                device=device,
                                label_ratio=args.label,
                                use_mean=args.use_mean)

  discriminator = GAILDiscrim(state_size=state_size,
                              action_size=action_size,
                              hidden_units=(100, 100),
                              hidden_activation=nn.Tanh()).to(device)

  buffer = RolloutBuffer(buffer_size=args.rollout_length,
                         state_size=state_size,
                         latent_size=lat_conf.latent_size,
                         action_size=action_size,
                         discrete_state=discrete_state,
                         discrete_latent=lat_conf.discrete_latent,
                         discrete_action=discrete_action,
                         device=device)

  policy_class = DiscretePolicy if discrete_action else ContinousPolicy
  actor = policy_class(state_size=state_size,
                       latent_size=lat_conf.latent_size,
                       action_size=action_size,
                       hidden_units=(64, 64),
                       hidden_activation=nn.Tanh()).to(device)
  critic = StateFunction(state_size=state_size,
                         latent_size=lat_conf.latent_size,
                         hidden_units=(64, 64),
                         hidden_activation=nn.Tanh()).to(device)
  trans_class = (DiscreteTransition
                 if lat_conf.discrete_latent else ContinousTransition)
  trans = trans_class(state_size=state_size,
                      latent_size=lat_conf.latent_size,
                      action_size=action_size,
                      hidden_units=(64, 64),
                      hidden_activation=nn.Tanh()).to(device)

  if args.algo not in ['digail']:
    raise NotImplementedError

  algo = ALGOS[args.algo](state_size=state_size,
                          latent_size=lat_conf.latent_size,
                          action_size=action_size,
                          discrete_state=discrete_state,
                          discrete_latent=lat_conf.discrete_latent,
                          discrete_action=discrete_action,
                          buffer_exp=buffer_exp,
                          discriminator=discriminator,
                          buffer=buffer,
                          actor=actor,
                          critic=critic,
                          transition=trans,
                          cb_init_latent=lat_conf.get_init_latent,
                          device=device,
                          seed=args.seed,
                          rollout_length=args.rollout_length)

  time = datetime.now().strftime("%Y%m%d-%H%M%S")
  log_dir = os.path.join('logs', args.env_id, args.algo,
                         f'seed{args.seed}-{time}')

  trainer = Trainer(env=env,
                    env_test=env_test,
                    algo=algo,
                    log_dir=log_dir,
                    num_steps=args.num_steps,
                    num_pretrain_steps=args.num_pretrain_steps,
                    eval_interval=args.eval_interval,
                    num_eval_episodes=args.num_eval_epi,
                    seed=args.seed)
  trainer.pretrain()
  trainer.train()


if __name__ == '__main__':
  p = argparse.ArgumentParser()

  # required
  p.add_argument('--buffer',
                 type=str,
                 required=True,
                 help='path to the demonstration buffer')
  p.add_argument('--env-id',
                 type=str,
                 required=True,
                 help='name of the environment')
  p.add_argument('--algo',
                 type=str,
                 required=True,
                 help='Imitation Learning algorithm to be trained')

  # custom
  p.add_argument('--rollout-length',
                 type=int,
                 default=10000,
                 help='rollout length of the buffer')
  p.add_argument('--num-steps',
                 type=int,
                 default=10**6,
                 help='number of steps to train')
  p.add_argument('--eval-interval',
                 type=int,
                 default=10**4,
                 help='time interval between evaluations')

  p.add_argument('--num-pretrain-steps',
                 type=int,
                 default=20000000,
                 help='pre-train steps')
  p.add_argument('--use-mean',
                 action='store_true',
                 default=False,
                 help='use state transition reward')

  # default
  p.add_argument('--num-eval-epi',
                 type=int,
                 default=5,
                 help='number of episodes for evaluation')
  p.add_argument('--seed', type=int, default=0, help='random seed')
  p.add_argument('--label',
                 type=float,
                 default=0.05,
                 help='ratio of labeled data')

  args = p.parse_args()
  run(args)
