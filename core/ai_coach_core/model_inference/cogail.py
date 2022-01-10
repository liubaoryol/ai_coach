from types import SimpleNamespace
import tqdm
import numpy as np
import random
import torch
import external.cogail.algo.gail as cogail_gail
import external.cogail.algo as cogail_algo
import external.cogail.model as cogail_model
import external.cogail.storage as cogail_storage
import external.gail_common_utils.utils as gail_utils
import external.gail_common_utils.envs as gail_env
import gym_aicoach  # noqa: F401
import ai_coach_core.models.mdp as mdp_lib


def cogail_w_ppo(mdp: mdp_lib.MDP,
                 num_latent,
                 possible_init_states,
                 sa_trajectories_no_terminal,
                 demo_batch_size=64,
                 ppo_batch_size=32,
                 n_steps=64,
                 total_timesteps=32000,
                 do_pretrain=True,
                 callback_policy=None):

  assert len(sa_trajectories_no_terminal) != 0

  args = SimpleNamespace()
  args.seed = 1  # random seed (default: 1)
  args.cuda = True
  args.cuda_deterministic = False  # sets flags for determinism when using CUDA (potentially slow!) # noqa: E501
  args.ppo_clip_param = 0.2  # ppo clip parameter (default: 0.2)
  args.ppo_epoch = 10  # number of ppo epochs (default: 4)
  args.ppo_num_mini_batch = ppo_batch_size  # number of batches for ppo (default: 32) # noqa: E501
  args.ppo_value_loss_coef = 0.5  # value loss coefficient (default: 0.5)
  args.ppo_entropy_coef = 0.01  # entropy term coefficient (default: 0.01)
  args.ppo_lr = 7e-4  # learning rate (default: 7e-4)
  args.ppo_eps = 1e-5  # RMSprop optimizer epsilon (default: 1e-5)
  args.ppo_max_grad_norm = 0.5  # max norm of gradients (default: 0.5)
  args.gail_batch_size = demo_batch_size  # gail batch size (default: 128)
  args.num_steps = n_steps  # number of forward steps in A2C (default: 5)
  args.num_processes = 1  # how many training CPU processes to use (default: 16)
  args.num_env_steps = total_timesteps  # number of environment steps to train (default: 10e6) # noqa: E501
  args.use_linear_lr_decay = False  # use a linear schedule on the learning rate
  args.gail_epoch = 5  # gail epochs (default: 5)
  args.use_gae = False  # generalized advantage estimation
  args.gamma = 0.95  # discount factor for rewards (default: 0.99)
  args.gae_lambda = 0.95  # gae lambda parameter (default: 0.95)
  args.use_proper_time_limits = False  # compute returns taking into account time limits # noqa: E501
  args.discr_hidden_dim = 100
  args.bc_pretrain_steps = 1

  # ---------- seed for random ----------
  torch.manual_seed(args.seed)
  torch.cuda.manual_seed_all(args.seed)
  # np.random.seed(args.seed)
  # random.seed(args.seed)

  if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

  torch.set_num_threads(1)
  device = torch.device("cuda:0" if args.cuda else "cpu")

  # ---------- create vec env from mdp ----------
  env_kwargs = dict(mdp=mdp, possible_init_states=possible_init_states)

  venv = gail_env.make_vec_envs('envfrommdp-v0',
                                seed=args.seed,
                                num_processes=args.num_processes,
                                gamma=args.gamma,
                                log_dir=None,
                                device=device,
                                allow_early_resets=False,
                                env_make_kwargs=env_kwargs)

  tuple_num_actions = tuple(mdp.list_num_actions)
  # ---------- policy net ----------
  # assumed observation space is discrete
  actor_critic = cogail_model.Policy(venv.observation_space,
                                     venv.action_space,
                                     num_latent,
                                     base_kwargs={'recurrent': False})
  actor_critic.to(device)

  # ---------- policy learner ----------
  agent = cogail_algo.PPO(actor_critic,
                          args.ppo_clip_param,
                          args.ppo_epoch,
                          args.ppo_num_mini_batch,
                          args.ppo_value_loss_coef,
                          args.ppo_entropy_coef,
                          lr=args.ppo_lr,
                          eps=args.ppo_eps,
                          max_grad_norm=args.ppo_max_grad_norm)

  # ---------- gail discriminator ----------
  discriminator = cogail_gail.Discriminator(num_obs=venv.observation_space.n,
                                            tuple_num_actions=tuple_num_actions,
                                            hidden_dim=args.discr_hidden_dim,
                                            device=device,
                                            use_ce=False)

  # ---------- set data loader ----------
  # TODO: need to convert this to compatible with my data
  expert_data = gail_utils.TorchDatasetConverter(sa_trajectories_no_terminal)
  drop_last = len(expert_data) > args.gail_batch_size
  gail_train_loader = torch.utils.data.DataLoader(
      dataset=expert_data,
      batch_size=args.gail_batch_size,
      shuffle=True,
      drop_last=drop_last)

  # NOTE: don't know what it is ... num_processes? recurrent_hidden_state_size?
  # NOTE: I guess it's just to store state, action, and any other info
  # that are happening at each step to log and to use for next step
  rollouts = cogail_storage.RolloutStorage(
      args.num_steps, args.num_processes, venv.observation_space, num_latent,
      venv.action_space, actor_critic.recurrent_hidden_state_size)

  # reset env
  obs = venv.reset()
  random_seed = random.randrange(num_latent)
  random_seed = torch.Tensor([[random_seed]]).long()
  random_seed = gail_utils.conv_discrete_2_onehot(random_seed, num_latent)

  rollouts.obs[0].copy_(obs)
  rollouts.random_seed[0].copy_(random_seed)
  rollouts.to(device)

  if do_pretrain:
    for j in range(args.bc_pretrain_steps):
      loss = agent.pretrain(gail_train_loader, device)
      print("Pretrain round {0}: loss {1}".format(j, loss))

    # policy after bc
    if callback_policy is not None:
      callback_policy(
          get_np_policy(actor_critic, num_latent, mdp.num_states,
                        tuple_num_actions, device))

  # episode_rewards = deque(maxlen=10)  # I think this is just for info
  # start = time.time()
  # ---------- training ----------
  num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes
  for j in tqdm.tqdm(range(num_updates)):
    if args.use_linear_lr_decay:
      gail_utils.update_linear_schedule(agent.optimizer, j, num_updates,
                                        args.ppo_lr)

    # !!!!! collect rollouts by alternating policy and env !!!!!
    for step in range(args.num_steps):
      # Sample actions
      # NOTE: recurrent_hidden_state, masks -- don't need for my domain.
      with torch.no_grad():
        (value, action, action_log_prob,
         recurrent_hidden_states) = actor_critic.act(
             rollouts.obs[step], rollouts.random_seed[step],
             rollouts.recurrent_hidden_states[step], rollouts.masks[step])

      # Obser reward and next obs
      # NOTE: info? does info include 'bad_transition' and 'episode' keyword?
      obs, reward, done, infos = venv.step(action)
      if done:
        random_seed = random.randrange(num_latent)
        random_seed = torch.Tensor([[random_seed]]).long()
        random_seed = gail_utils.conv_discrete_2_onehot(random_seed, num_latent)

      # NOTE: what's this for?
      # for info in infos:
      #   if 'episode' in info.keys():
      #     episode_rewards.append(info['episode']['r'])

      # If done then clean the history of observations. (?????)
      # NOTE: FloatTensor..?
      masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
      bad_masks = torch.FloatTensor(
          [[0.0] if 'bad_transition' in info.keys() else [1.0]
           for info in infos])
      rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob,
                      value, reward, masks, bad_masks, random_seed)

    # !!!!! after collecting rollouts !!!!!
    with torch.no_grad():
      next_value = actor_critic.get_value(rollouts.obs[-1],
                                          rollouts.random_seed[-1],
                                          rollouts.recurrent_hidden_states[-1],
                                          rollouts.masks[-1]).detach()

    # NOTE: don't know what this is. I feel this is not needed.
    # if j >= 10:
    #   venv.venv.eval()

    gail_epoch = args.gail_epoch
    # if j < 10:
    #   gail_epoch = 100  # Warm up

    # !!!!! gail update !!!!!
    # discriminate expert data (gail_train_loader) and generated data (rollouts)
    # NOTE: don't know what obfilt is.
    for _ in range(gail_epoch):
      # NOTE: need to change update function to work with discrete spaces
      discriminator.update(gail_train_loader, rollouts)
      #  gail_utils.get_vec_normalize(venv)._obfilt)

    # !!!!! predict rewards from discriminator !!!!!
    for step in range(args.num_steps):
      rollouts.rewards[step] = discriminator.predict_reward(
          rollouts.obs[step], rollouts.actions[step],
          rollouts.random_seed[step], args.gamma, rollouts.masks[step])

    # NOTE: what are returns for?
    # NOTE: what's the meaning of each paramter?
    rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                             args.gae_lambda, args.use_proper_time_limits)

    # !!!!! udpate agent with rewards(return?) from discriminator !!!!!
    # NOTE: need to change update function to work with discrete spaces
    value_loss, action_loss, dist_entropy, code_loss, inv_loss = agent.update(
        rollouts, gail_train_loader, device)

    rollouts.after_update()

    if callback_policy is not None:
      callback_policy(
          get_np_policy(actor_critic, num_latent, mdp.num_states,
                        tuple_num_actions, device))

  return get_np_policy(actor_critic, num_latent, mdp.num_states,
                       tuple_num_actions, device)


def get_np_policy(actor_critic: cogail_model.Policy, num_code: int,
                  num_states: int, tuple_num_actions: tuple, device):
  list_np_policy = [
      np.zeros((num_code, num_states, tuple_num_actions[idx]))
      for idx in range(2)
  ]

  # to boost speed, compute probs by batch
  # --> needed to change max memory depending on machine
  MAX_MEMORY = 536870912  # 512MiB
  batch_size = int((MAX_MEMORY / 8) / num_states)  # 8Byte = double type

  with torch.no_grad():
    for xidx in range(num_code):
      code = torch.Tensor([[xidx]]).long()
      for batch_idx in range(0, num_states, batch_size):
        end_idx = min(batch_idx + batch_size, num_states)

        state_input = torch.Tensor(
            np.arange(batch_idx, end_idx).reshape(-1, 1)).long()
        state_input = state_input.to(device)

        code_1hot = gail_utils.conv_discrete_2_onehot(code, num_code).to(device)
        code_1hot = code_1hot.expand(end_idx - batch_idx, num_code)

        dist1, dist2 = actor_critic.get_distribution(state_input, code_1hot,
                                                     None, None)
        probs1 = dist1.probs.cpu().numpy()
        probs2 = dist2.probs.cpu().numpy()

        list_np_policy[0][xidx, batch_idx:end_idx] = probs1
        list_np_policy[1][xidx, batch_idx:end_idx] = probs2

  return list_np_policy
