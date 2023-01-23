from typing import Sequence, Tuple, Callable, Optional
import numpy as np
from scipy.special import logsumexp, softmax

# type for a callback that takes the input: (agent_idx, latent_idx, state_idx,
#                                            tuple of a team's action_idx,
#                                            next_state_idx, next_latent_idx)
T_CallbackLatentTransition = Callable[
    [int, int, int, Tuple[int, ...], int, int], float]

# type for a policy that takes the input: (agent_idx, latent_idx, state_idx,
#                                          tuple of a team's action_idx)
T_CallbackPolicy = Callable[[int, int, int, Tuple[int, ...]], float]


def most_probable_sequence(state_seq: Sequence[int],
                           action_seq: Sequence[Tuple[int,
                                                      ...]], num_agents: int,
                           num_latent: int, cb_policy: T_CallbackPolicy,
                           cb_transition_x: T_CallbackLatentTransition,
                           cb_prior: Callable[[int, int, int], float]):
  '''
  state_seq: trajectory of the state
  action_seq: trajectory of joint actions.
  num_agents: the number of agents that can individually have a mental state
  '''
  num_step = len(action_seq)

  log_probs = []  # type: list[np.ndarray]
  log_trackers = []  # type: list[list[list[int]]]
  for dummy in range(num_agents):
    log_probs.append(np.zeros((num_step, num_latent)))
    log_trackers.append([])

  s_0 = state_seq[0]
  tuple_action = action_seq[0]
  for idx in range(num_agents):
    for x_i in range(num_latent):
      p_act_sum = cb_policy(idx, x_i, s_0, tuple_action)

      p_x_sum = cb_prior(idx, x_i, s_0)

      if p_act_sum == 0 or p_x_sum == 0:
        log_probs[idx][0, x_i] = -np.inf
      else:
        log_probs[idx][0, x_i] = np.log(p_act_sum) + np.log(p_x_sum)
      log_trackers[idx].append([x_i])

  for step in range(1, num_step):
    s_i = state_seq[step]
    tuple_action_i = action_seq[step]
    s_j = state_seq[step - 1]
    tuple_action_j = action_seq[step - 1]
    for idx in range(num_agents):
      prev_tracks_log = log_trackers[idx]
      new_tracks_log = []
      for x_i in range(num_latent):
        p_act_sum = cb_policy(idx, x_i, s_i, tuple_action_i)

        max_idx_log = -1
        max_log_val = -np.inf
        for x_j in range(num_latent):
          p_x_n_sum = cb_transition_x(idx, x_j, s_j, tuple_action_j, s_i, x_i)

          if p_x_n_sum == 0:
            log_p_tmp = -np.inf
          else:
            log_p_tmp = log_probs[idx][step - 1, x_j] + np.log(p_x_n_sum)

          if log_p_tmp > max_log_val:
            max_idx_log = x_j
            max_log_val = log_p_tmp

        if p_act_sum == 0:
          log_probs[idx][step, x_i] = -np.inf
        else:
          log_probs[idx][step, x_i] = np.log(p_act_sum) + max_log_val
        best_track_log = list(prev_tracks_log[max_idx_log])
        best_track_log.append(x_i)
        new_tracks_log.append(best_track_log)
      log_trackers[idx] = new_tracks_log

  log_sequences = []
  for idx in range(num_agents):
    track_idx_log = np.argmax(log_probs[idx][-1, :])
    log_sequences.append(log_trackers[idx][track_idx_log])

  return log_sequences


def forward_inference(state_seq: Sequence[int],
                      action_seq: Sequence[Tuple[int, ...]],
                      num_agents: int,
                      num_latent: int,
                      cb_policy: T_CallbackPolicy,
                      cb_transition_x: T_CallbackLatentTransition,
                      cb_prior: Callable[[int, int, int], float],
                      list_np_prev_px: Optional[Sequence[np.ndarray]] = None):
  '''
  cb_prev_px: takes agent_idx as input and
              returns the distribution of x at the previous step
  '''

  assert len(state_seq) == len(action_seq) + 1
  assert len(state_seq) > 0

  list_np_px = []
  list_max_x = []

  # if previous p(x'|x, s, a, s') is given
  if list_np_prev_px is not None:
    t_end = len(state_seq) - 1
    s_t = state_seq[t_end]
    s_tp = state_seq[t_end - 1]
    joint_a_tp = action_seq[t_end - 1]
    for idx in range(num_agents):
      np_px = np.zeros(num_latent)
      for x_t in range(num_latent):
        sum_px = 0.0
        for x_tp in range(num_latent):
          sum_px += (list_np_prev_px[idx][x_tp] *
                     cb_transition_x(idx, x_tp, s_tp, joint_a_tp, s_t, x_t) *
                     cb_policy(idx, x_tp, s_tp, joint_a_tp))
        np_px[x_t] = sum_px
      np_px /= np.sum(np_px)

      list_np_px.append(np_px)
      list_max_x.append(np.argmax(np_px))

    return list_max_x, list_np_px
  else:
    s0 = state_seq[0]
    for idx in range(num_agents):
      np_forward = np.zeros((len(state_seq), num_latent))
      np_forward[0, :] = [cb_prior(idx, xidx, s0) for xidx in range(num_latent)]
      for step in range(1, len(state_seq)):
        s_t = state_seq[step]
        s_tp = state_seq[step - 1]
        joint_a_tp = action_seq[step - 1]
        for x_t in range(num_latent):
          sum_px = 0.0
          for x_tp in range(num_latent):
            sum_px += (np_forward[step - 1, x_tp] *
                       cb_transition_x(idx, x_tp, s_tp, joint_a_tp, s_t, x_t) *
                       cb_policy(idx, x_tp, s_tp, joint_a_tp))
          np_forward[step, x_t] = sum_px
        np_forward[step, :] = np_forward[step, :] / np.sum(np_forward[step, :])

      list_np_px.append(np_forward[-1, :])
      list_max_x.append(np.argmax(np_forward[-1, :]))

    return list_max_x, list_np_px


def smooth_inference_sa(state_seq: Sequence[int],
                        action_seq: Sequence[Tuple[int, ...]], num_agents: int,
                        tuple_num_latent: Sequence[int],
                        list_np_pi: Sequence[np.ndarray],
                        list_np_tx: Sequence[np.ndarray],
                        list_np_bx: Sequence[np.ndarray]):

  len_traj = len(action_seq)

  list_np_px = []
  for idx_a in range(num_agents):
    # Forward messaging
    with np.errstate(divide='ignore'):
      np_log_forward = np.log(np.zeros((len_traj, tuple_num_latent[idx_a])))

    t = 0
    state_p, joint_a_p = state_seq[t], action_seq[t]

    with np.errstate(divide='ignore'):
      np_log_forward[t] = 0.0
      np_log_forward[t] += np.log(list_np_bx[idx_a][state_p])
      np_log_forward[t] += np.log(list_np_pi[idx_a][:, state_p,
                                                    joint_a_p[idx_a]])
    # t = 1:N-1
    for t in range(1, len_traj):
      t_p = t - 1
      state, joint_a = state_seq[t], action_seq[t]

      with np.errstate(divide='ignore'):
        np_log_prob = np_log_forward[t_p][:, None]

        tx_index = (slice(None), *joint_a_p, state, slice(None))
        np_log_prob = np_log_prob + np.log(list_np_tx[idx_a][tx_index])
        np_log_prob = np_log_prob + np.log(
            list_np_pi[idx_a][:, state, joint_a[idx_a]])[None, :]
      np_log_forward[t] = logsumexp(np_log_prob, axis=0)

      joint_a_p = joint_a

    # Backward messaging
    with np.errstate(divide='ignore'):
      np_log_backward = np.log(np.zeros((len_traj, tuple_num_latent[idx_a])))
    # t = N-1
    t = len_traj - 1

    state_n, joint_a_n = state_seq[t], action_seq[t]

    np_log_backward[t] = 0.0

    # t = 0:N-2
    for t in reversed(range(0, len_traj - 1)):
      t_n = t + 1
      state, joint_a = state_seq[t], action_seq[t]

      with np.errstate(divide='ignore'):

        np_log_prob = np_log_backward[t_n][None, :]

        tx_index = (slice(None), *joint_a, state_n, slice(None))
        np_log_prob = np_log_prob + np.log(list_np_tx[idx_a][tx_index])
        np_log_prob = np_log_prob + np.log(
            list_np_pi[idx_a][:, state_n, joint_a_n[idx_a]])[None, :]
      np_log_backward[t] = logsumexp(np_log_prob, axis=1)

      joint_a_n = joint_a

    # compute q_zx
    log_q_x = np_log_forward + np_log_backward

    q_x = softmax(log_q_x, axis=1)
    list_np_px.append(q_x)

  return list_np_px


def smooth_inference_zx(state_seq: Sequence[int],
                        action_seq: Sequence[Tuple[int, ...]], num_agents: int,
                        tuple_num_latent: Sequence[int], num_abstate: int,
                        np_abs: np.ndarray, list_np_pi: Sequence[np.ndarray],
                        list_np_tx: Sequence[np.ndarray],
                        list_np_bx: Sequence[np.ndarray]):
  '''
  cb_prev_px: takes agent_idx as input and
              returns the distribution of x at the previous step
  '''

  len_traj = len(action_seq)

  # Forward messaging
  with np.errstate(divide='ignore'):
    np_log_forward = np.log(np.zeros(
        (len_traj, num_abstate, *tuple_num_latent)))

  t = 0
  state_p, joint_a_p = state_seq[t], action_seq[t]

  all_one = (1, ) * num_agents
  with np.errstate(divide='ignore'):
    np_log_forward[t] = 0.0
    np_log_forward[t] += np.log(np_abs[state_p]).reshape(-1, *all_one)
    for idx_a in range(num_agents):
      shape_tmp = [1] * num_agents
      shape_tmp[idx_a] = tuple_num_latent[idx_a]
      shape_tmp = tuple(shape_tmp)
      np_log_forward[t] += np.log(list_np_bx[idx_a]).reshape(-1, *shape_tmp)
      np_log_forward[t] += np.log(
          list_np_pi[idx_a][:, :, joint_a_p[idx_a]]).transpose().reshape(
              -1, *shape_tmp)

  # t = 1:N-1
  for t in range(1, len_traj):
    t_p = t - 1
    state, joint_a = state_seq[t], action_seq[t]

    with np.errstate(divide='ignore'):
      np_log_prob = np_log_forward[t_p].reshape(num_abstate, *tuple_num_latent,
                                                1, *all_one)
      np_log_prob = np_log_prob + np.log(np_abs[state]).reshape(
          1, *all_one, -1, *all_one)
      for idx_a in range(num_agents):
        shape_tmp = [1] * num_agents
        shape_tmp[idx_a] = tuple_num_latent[idx_a]
        shape_tmp = tuple(shape_tmp)
        tx_index = (slice(None), *joint_a_p, slice(None), slice(None))
        np_log_prob = np_log_prob + np.log(list_np_tx[idx_a][tx_index]).reshape(
            1, *shape_tmp, -1, *shape_tmp)
        np_log_prob = np_log_prob + np.log(
            list_np_pi[idx_a][:, :, joint_a[idx_a]]).transpose().reshape(
                1, *all_one, -1, *shape_tmp)

    np_log_forward[t] = logsumexp(np_log_prob,
                                  axis=tuple(range(num_agents + 1)))

    joint_a_p = joint_a

  # Backward messaging
  with np.errstate(divide='ignore'):
    np_log_backward = np.log(
        np.zeros((len_traj, num_abstate, *tuple_num_latent)))
  # t = N-1
  t = len_traj - 1

  state_n, joint_a_n = state_seq[t], action_seq[t]

  np_log_backward[t] = 0.0

  # t = 0:N-2
  for t in reversed(range(0, len_traj - 1)):
    t_n = t + 1
    _, joint_a = state_seq[t], action_seq[t]

    with np.errstate(divide='ignore'):

      np_log_prob = np_log_backward[t_n].reshape(1, *all_one, num_abstate,
                                                 *tuple_num_latent)
      np_log_prob = np_log_prob + np.log(np_abs[state_n]).reshape(
          1, *all_one, -1, *all_one)
      for idx_a in range(num_agents):
        shape_tmp = [1] * num_agents
        shape_tmp[idx_a] = tuple_num_latent[idx_a]
        shape_tmp = tuple(shape_tmp)
        tx_index = (slice(None), *joint_a, slice(None), slice(None))
        np_log_prob = np_log_prob + np.log(list_np_tx[idx_a][tx_index]).reshape(
            1, *shape_tmp, -1, *shape_tmp)
        np_log_prob = np_log_prob + np.log(
            list_np_pi[idx_a][:, :, joint_a_n[idx_a]]).transpose().reshape(
                1, *all_one, -1, *shape_tmp)

    np_log_backward[t] = logsumexp(np_log_prob,
                                   axis=tuple(
                                       range(num_agents + 1,
                                             2 * num_agents + 2)))  # noqa: E501

    joint_a_n = joint_a

  # compute q_zx
  log_q_zx = np_log_forward + np_log_backward

  q_zx = softmax(log_q_zx, axis=tuple(range(1, log_q_zx.ndim)))

  return q_zx


def smooth_inference_max_z(state_seq: Sequence[int],
                           action_seq: Sequence[Tuple[int,
                                                      ...]], num_agents: int,
                           tuple_num_latent: Sequence[int], num_abstate: int,
                           np_abs: np.ndarray, list_np_pi: Sequence[np.ndarray],
                           list_np_tx: Sequence[np.ndarray],
                           list_np_bx: Sequence[np.ndarray]):
  '''
  cb_prev_px: takes agent_idx as input and
              returns the distribution of x at the previous step
  '''

  list_z = []
  for s in state_seq:
    z = np.argmax(np_abs[s])
    list_z.append(z)

  len_traj = len(action_seq)

  list_qzx = [list_z]
  for idx_a in range(num_agents):
    # Forward messaging
    with np.errstate(divide='ignore'):
      np_log_forward = np.log(np.zeros((len_traj, tuple_num_latent[idx_a])))

    t = 0
    state_p, joint_a_p = list_z[t], action_seq[t]

    with np.errstate(divide='ignore'):
      np_log_forward[t] = 0.0
      np_log_forward[t] += np.log(list_np_bx[idx_a][state_p])
      np_log_forward[t] += np.log(list_np_pi[idx_a][:, state_p,
                                                    joint_a_p[idx_a]])
    # t = 1:N-1
    for t in range(1, len_traj):
      t_p = t - 1
      state, joint_a = list_z[t], action_seq[t]

      with np.errstate(divide='ignore'):
        np_log_prob = np_log_forward[t_p][:, None]

        tx_index = (slice(None), *joint_a_p, state, slice(None))
        np_log_prob = np_log_prob + np.log(list_np_tx[idx_a][tx_index])
        np_log_prob = np_log_prob + np.log(
            list_np_pi[idx_a][:, state, joint_a[idx_a]])[None, :]
      np_log_forward[t] = logsumexp(np_log_prob, axis=0)

      joint_a_p = joint_a

    # Backward messaging
    with np.errstate(divide='ignore'):
      np_log_backward = np.log(np.zeros((len_traj, tuple_num_latent[idx_a])))
    # t = N-1
    t = len_traj - 1

    state_n, joint_a_n = list_z[t], action_seq[t]

    np_log_backward[t] = 0.0

    # t = 0:N-2
    for t in reversed(range(0, len_traj - 1)):
      t_n = t + 1
      state, joint_a = list_z[t], action_seq[t]

      with np.errstate(divide='ignore'):

        np_log_prob = np_log_backward[t_n][None, :]

        tx_index = (slice(None), *joint_a, state_n, slice(None))
        np_log_prob = np_log_prob + np.log(list_np_tx[idx_a][tx_index])
        np_log_prob = np_log_prob + np.log(
            list_np_pi[idx_a][:, state_n, joint_a_n[idx_a]])[None, :]
      np_log_backward[t] = logsumexp(np_log_prob, axis=1)

      joint_a_n = joint_a

    # compute q_zx
    log_q_x = np_log_forward + np_log_backward

    q_x = softmax(log_q_x, axis=1)
    list_qzx.append(q_x)

  return list_qzx