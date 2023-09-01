from typing import Sequence
import numpy as np


def norm_hamming_distance(seq1, seq2):
  return hamming_distance(seq1, seq2) / len(seq1)


def hamming_distance(seq1, seq2):
  assert len(seq1) == len(seq2)

  count = 0
  for idx, elem in enumerate(seq1):
    if elem != seq2[idx]:
      count += 1

  return count


def alignment_sequence(seq1, seq2):
  assert len(seq1) == len(seq2)

  seq_align = []
  for idx in range(len(seq1)):
    if seq1[idx] == seq2[idx]:
      seq_align.append(1)
    else:
      seq_align.append(0)

  return seq_align


def compute_relative_freq(num_agents: int, num_latents: int, num_states: int,
                          trajs: Sequence[Sequence]):
  list_rel_freq = [
      np.zeros((num_latents, num_states)) for _ in range(num_agents)
  ]

  count = 0
  for traj in trajs:
    for s, _, joint_x in traj:
      if joint_x[0] is None or joint_x[1] is None:
        continue
      for idx in range(num_agents):
        list_rel_freq[idx][joint_x[idx], s] += 1
      count += 1

  for idx in range(num_agents):
    list_rel_freq[idx] = list_rel_freq[idx] / count

  return list_rel_freq


def compute_kl(np_p, np_q):
  sum_val = 0
  for idx in range(len(np_p)):
    if np_p[idx] != 0 and np_q[idx] != 0:
      sum_val += np_p[idx] * (np.log(np_p[idx]) - np.log(np_q[idx]))
  return sum_val


def compute_js(np_p, np_q):
  np_m = 0.5 * np_p + 0.5 * np_q
  return 0.5 * compute_kl(np_p, np_m) + 0.5 * compute_kl(np_q, np_m)


def compute_01(np_p, np_q):
  idxmax_p = np.where(np_p == np.max(np_p))
  idxmax_q = np.where(np_q == np.max(np_q))

  intersect = np.intersect1d(idxmax_p, idxmax_q)

  return 1 if len(intersect) > 0 else 0


def cal_latent_policy_error(num_agents, num_states, num_latents,
                            sax_trajectories, cb_pi_true, cb_pi_infer):

  list_rel_freq = compute_relative_freq(num_agents, num_latents, num_states,
                                        sax_trajectories)

  dict_results = {}
  dict_results['wKL'] = []
  dict_results['KL'] = []
  dict_results['wJS'] = []
  dict_results['JS'] = []
  dict_results['w01'] = []
  dict_results['01'] = []
  for aidx in range(num_agents):
    sum_wkl, sum_kl, sum_wjs, sum_js, sum_w01, sum_01 = 0, 0, 0, 0, 0, 0
    for xidx in range(num_latents):
      for sidx in range(num_states):
        weight = list_rel_freq[aidx][xidx, sidx]
        np_p_inf = cb_pi_infer(aidx, xidx, sidx)
        np_p_true = cb_pi_true(aidx, xidx, sidx)

        d_kl = compute_kl(np_p_inf, np_p_true)
        d_js = compute_js(np_p_inf, np_p_true)

        # inf_act = np.argmax(np_p_inf)
        # true_act = np.argmax(np_p_true)

        sum_kl += d_kl
        sum_js += d_js

        # sum_01 += 1 if inf_act == true_act else 0
        num_01 = compute_01(np_p_inf, np_p_true)
        sum_01 += num_01

        if weight > 0:
          sum_wkl += weight * d_kl
          sum_wjs += weight * d_js
          sum_w01 += weight * num_01

    count = num_latents * num_states

    dict_results['wKL'].append(sum_wkl)
    dict_results['KL'].append(sum_kl / count)
    dict_results['wJS'].append(sum_wjs)
    dict_results['JS'].append(sum_js / count)
    dict_results['w01'].append(1 - sum_w01)
    dict_results['01'].append(1 - sum_01 / count)

  return dict_results
