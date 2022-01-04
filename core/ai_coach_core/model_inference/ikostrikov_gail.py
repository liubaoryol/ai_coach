from types import SimpleNamespace
import tqdm
import numpy as np
import torch
import external.a2c_ppo_acktr.algo.gail as ikostrikov_gail
import external.a2c_ppo_acktr.algo as ikostrikov_algo
import external.a2c_ppo_acktr.model as ikostrikov_model
import external.a2c_ppo_acktr.storage as ikostrikov_storage
import external.a2c_ppo_acktr.utils as ikostrikov_utils
import external.a2c_ppo_acktr.envs as ikostrikov_env
import gym_aicoach  # noqa: F401
import ai_coach_core.models.mdp as mdp_lib
from ai_coach_core.utils.data_utils import TorchDatasetConverter


def gail_w_ppo(mdp: mdp_lib.MDP,
               init_state,
               sa_trajectories_no_terminal,
               demo_batch_size=64,
               ppo_batch_size=32,
               n_steps=64,
               total_timesteps=32000,
               do_pretrain=True,
               only_pretrain=False,
               callback_policy=None):

  args = SimpleNamespace()
  args.seed = 1  # random seed (default: 1)
  args.cuda = True
  args.cuda_deterministic = False  # sets flags for determinism when using CUDA (potentially slow!)  # noqa: E501
  args.ppo_clip_param = 0.2  # ppo clip parameter (default: 0.2)
  args.ppo_epoch = 10  # number of ppo epochs (default: 4)
  args.ppo_num_mini_batch = ppo_batch_size  # number of batches for ppo (default: 32) # noqa: E501
  args.ppo_value_loss_coef = 0.5  # value loss coefficient (default: 0.5)
  args.ppo_entropy_coef = 0.01  # entropy term coefficient (default: 0.01)
  args.ppo_lr = 1e-3  # learning rate (default: 7e-4)
  args.ppo_eps = 1e-8  # RMSprop optimizer epsilon (default: 1e-5)
  args.ppo_max_grad_norm = 0.5  # max norm of gradients (default: 0.5)
  args.gail_batch_size = demo_batch_size  # gail batch size (default: 128)
  args.num_steps = n_steps  # number of forward steps in A2C (default: 5)
  args.num_processes = 1  # how many training CPU processes to use (default: 16)
  args.num_env_steps = total_timesteps  # number of environment steps to train (default: 10e6) # noqa: E501
  args.use_linear_lr_decay = False  # use a linear schedule on the learning rate
  args.gail_epoch = 2  # gail epochs (default: 5)
  args.use_gae = False  # generalized advantage estimation
  args.gamma = 0.99  # discount factor for rewards (default: 0.99)
  args.gae_lambda = 0.95  # gae lambda parameter (default: 0.95)
  args.use_proper_time_limits = False  # compute returns taking into account time limits # noqa: E501
  args.discr_hidden_dim = 100
  args.bc_pretrain_steps = 1

  # ---------- torch settings ----------
  # torch.manual_seed(args.seed)
  # torch.cuda.manual_seed_all(args.seed)

  if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

  torch.set_num_threads(1)
  device = torch.device("cuda:0" if args.cuda else "cpu")

  # ---------- create vec env from mdp ----------
  # gymenv = GymEnvFromMDP(mdp.num_states, mdp.num_actions, mdp.transition,
  #                        mdp.is_terminal, mdp.legal_actions, init_state)
  # venv = DummyVecEnv([lambda: gymenv])

  env_kwargs = dict(num_states=mdp.num_states,
                    num_actions=mdp.num_actions,
                    cb_transition=mdp.transition,
                    cb_is_terminal=mdp.is_terminal,
                    cb_is_legal_action=lambda s, a: a in mdp.legal_actions(s),
                    init_state=init_state)

  venv = ikostrikov_env.make_vec_envs('envfrommdp-v0',
                                      seed=args.seed,
                                      num_processes=args.num_processes,
                                      gamma=args.gamma,
                                      log_dir='',
                                      device=device,
                                      allow_early_resets=False,
                                      env_make_kwargs=env_kwargs)

  # ---------- policy net ----------
  # assumed observation space is discrete
  actor_critic = ikostrikov_model.Policy(venv.observation_space,
                                         venv.action_space,
                                         base_kwargs={'recurrent': False})
  actor_critic.to(device)

  # ---------- policy learner ----------
  agent = ikostrikov_algo.PPO(actor_critic,
                              args.ppo_clip_param,
                              args.ppo_epoch,
                              args.ppo_num_mini_batch,
                              args.ppo_value_loss_coef,
                              args.ppo_entropy_coef,
                              lr=args.ppo_lr,
                              eps=args.ppo_eps,
                              max_grad_norm=args.ppo_max_grad_norm)

  # ---------- gail discriminator ----------
  discriminator = ikostrikov_gail.Discriminator(
      input_dim=(venv.observation_space.n + venv.action_space.n),
      hidden_dim=args.discr_hidden_dim,
      device=device,
      is_discrete_obs=True,
      is_discrete_action=True,
      num_obs=venv.observation_space.n,
      num_action=venv.action_space.n)

  # ---------- set data loader ----------
  expert_data = TorchDatasetConverter(sa_trajectories_no_terminal)
  drop_last = len(expert_data) > args.gail_batch_size
  gail_train_loader = torch.utils.data.DataLoader(
      dataset=expert_data,
      batch_size=args.gail_batch_size,
      shuffle=True,
      drop_last=drop_last)

  # rollout storage
  rollouts = ikostrikov_storage.RolloutStorage(
      args.num_steps, args.num_processes, venv.observation_space,
      venv.action_space, actor_critic.recurrent_hidden_state_size)

  # reset env
  obs = venv.reset()

  # NOTE: don't know what is this for...
  rollouts.obs[0].copy_(obs)
  rollouts.to(device)

  if do_pretrain:
    for j in range(args.bc_pretrain_steps):
      loss = agent.pretrain(gail_train_loader, device)
      print("Pretrain round {0}: loss {1}".format(j, loss))

    if only_pretrain:
      return get_np_policy(actor_critic, mdp.num_states, mdp.num_actions,
                           device)

    # policy after bc
    if callback_policy is not None:
      callback_policy(
          get_np_policy(actor_critic, mdp.num_states, mdp.num_actions, device))

  # ---------- training ----------
  num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes
  for j in tqdm.tqdm(range(num_updates)):
    if args.use_linear_lr_decay:
      ikostrikov_utils.update_linear_schedule(agent.optimizer, j, num_updates,
                                              args.ppo_lr)

    # ...... collect rollouts by alternating policy and env ......
    for step in range(args.num_steps):
      # Sample actions
      with torch.no_grad():
        (value, action, action_log_prob,
         recurrent_hidden_states) = actor_critic.act(
             rollouts.obs[step], rollouts.recurrent_hidden_states[step],
             rollouts.masks[step])

      # Obser reward and next obs
      # NOTE: info? does info include 'bad_transition' and 'episode' keyword?
      obs, reward, done, infos = venv.step(action)

      # ??
      masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
      bad_masks = torch.FloatTensor(
          [[0.0] if 'bad_transition' in info.keys() else [1.0]
           for info in infos])

      rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob,
                      value, reward, masks, bad_masks)

    # ...... after collecting rollouts ......
    with torch.no_grad():
      next_value = actor_critic.get_value(rollouts.obs[-1],
                                          rollouts.recurrent_hidden_states[-1],
                                          rollouts.masks[-1]).detach()

    gail_epoch = args.gail_epoch
    # if j < 10:
    #   gail_epoch = 100  # Warm up

    # ...... gail update ......
    # discriminate expert data(gail_train_loader) and generated data (rollouts)
    # NOTE: don't know what obfilt is.
    for _ in range(gail_epoch):
      # NOTE: need to change update function to work with discrete spaces
      discriminator.update(gail_train_loader, rollouts)

    # ...... predict rewards from discriminator ......
    for step in range(args.num_steps):
      rollouts.rewards[step] = discriminator.predict_reward(
          rollouts.obs[step], rollouts.actions[step], args.gamma,
          rollouts.masks[step])

    # NOTE: what are returns for?
    # NOTE: what's the meaning of each paramter?
    rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                             args.gae_lambda, args.use_proper_time_limits)

    # ...... udpate agent with rewards(return?) from discriminator ......
    value_loss, action_loss, dist_entropy = agent.update(rollouts)

    rollouts.after_update()

    if callback_policy is not None:
      callback_policy(
          get_np_policy(actor_critic, mdp.num_states, mdp.num_actions, device))

  return get_np_policy(actor_critic, mdp.num_states, mdp.num_actions, device)


def get_np_policy(actor_critic: ikostrikov_model.Policy, num_states: int,
                  num_actions: int, device):
  np_policy = np.zeros((num_states, num_actions))
  with torch.no_grad():
    for sidx in range(num_states):
      state_input = torch.Tensor([[sidx]]).long()
      state_input = state_input.to(device)
      dist = actor_critic.get_distribution(state_input, None, None)
      probs = dist.probs.cpu().numpy()
      np_policy[sidx, :] = probs

  return np_policy
