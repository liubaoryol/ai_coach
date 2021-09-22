from typing import Sequence, Tuple, Callable
import numpy as np
from ai_coach_core.models.mdp import MDP


def most_probable_sequence(
    mdp: MDP,
    state_seq: Sequence[int],
    action_seq: Sequence[Tuple[int, ...]],
    num_minds: int,
    num_latent: int,
    cb_policy: Callable[[int, int, int, Tuple[int, ...]], float],
    cb_transition: Callable[[int, int, int, int, int], float],
    cb_prior: Callable[[int, int], float],
):
  num_step = len(state_seq)

  # probs = []
  # trackers = []  # type: list[list[int]]
  log_probs = []  # type: list[np.ndarray]
  log_trackers = []  # type: list[list[list[int]]]
  for dummy in range(num_minds):
    # probs.append(np.zeros((num_step, num_latent)))
    # trackers.append([])
    log_probs.append(np.zeros((num_step, num_latent)))
    log_trackers.append([])

  s_0 = state_seq[0]
  tuple_action = action_seq[0]
  for idx in range(num_minds):
    for x_i in range(num_latent):
      p_act_sum = cb_policy(idx, s_0, x_i, tuple_action)

      p_x_sum = cb_prior(idx, x_i)

      # probs[idx][0, x_i] = p_act_sum * p_x_sum
      # trackers[idx].append([x_i])
      log_probs[idx][0, x_i] = np.log(p_act_sum) + np.log(p_x_sum)
      log_trackers[idx].append([x_i])

  for step in range(1, num_step):
    s_i = state_seq[step]
    tuple_action_i = action_seq[step]
    s_j = state_seq[step - 1]
    tuple_action_j = action_seq[step - 1]
    joint_a_j = mdp.np_action_to_idx[tuple_action_j]
    for idx in range(num_minds):
      # prev_tracks = trackers[idx]
      # new_tracks = []
      prev_tracks_log = log_trackers[idx]
      new_tracks_log = []
      for x_i in range(num_latent):
        p_act_sum = cb_policy(idx, s_i, x_i, tuple_action_i)

        # max_idx = -1
        # max_val = -np.inf
        max_idx_log = -1
        max_log_val = -np.inf
        for x_j in range(num_latent):
          p_x_n_sum = cb_transition(idx, s_j, joint_a_j, x_j, x_i)

          # p_tmp = probs[idx][step - 1, x_j] * p_x_n_sum
          # if p_tmp > max_val:
          #     max_idx = x_j
          #     max_val = p_tmp
          log_p_tmp = log_probs[idx][step - 1, x_j] + np.log(p_x_n_sum)
          if log_p_tmp > max_log_val:
            max_idx_log = x_j
            max_log_val = log_p_tmp

        # probs[idx][step, x_i] = p_act_sum * max_val
        # best_track = list(prev_tracks[max_idx])
        # best_track.append(x_i)
        # new_tracks.append(best_track)
        log_probs[idx][step, x_i] = np.log(p_act_sum) + max_log_val
        best_track_log = list(prev_tracks_log[max_idx_log])
        best_track_log.append(x_i)
        new_tracks_log.append(best_track_log)
      # trackers[idx] = new_tracks
      log_trackers[idx] = new_tracks_log

  # print(probs)
  # print(log_probs)
  # sequences = []
  log_sequences = []
  for idx in range(num_minds):
    # track_idx = np.argmax(probs[idx][-1, :])
    # sequences.append(trackers[idx][track_idx])
    track_idx_log = np.argmax(log_probs[idx][-1, :])
    log_sequences.append(log_trackers[idx][track_idx_log])

  # print(log_sequences)

  return log_sequences


def norm_hamming_distance(seq1, seq2):
  assert len(seq1) == len(seq2)

  count = 0
  for idx, elem in enumerate(seq1):
    if elem != seq2[idx]:
      count += 1

  return count / len(seq1)


def alignment_sequence(seq1, seq2):
  assert len(seq1) == len(seq2)

  seq_align = []
  for idx in range(len(seq1)):
    if seq1[idx] == seq2[idx]:
      seq_align.append(1)
    else:
      seq_align.append(0)

  return seq_align
