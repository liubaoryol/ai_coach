from typing import Sequence, Tuple, Callable, Optional
import numpy as np

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

  list_np_px = []
  list_max_x = []
  # if previous p(x'|x, s, a, s') is given
  if list_np_prev_px is not None:
    s_t = state_seq[-1]
    s_tp = state_seq[-2]
    joint_a_tp = state_seq[-1]
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
