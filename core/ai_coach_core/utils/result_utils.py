import numpy as np


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


def cal_policy_kl_error(num_agents, num_states, num_latents, sax_trajectories,
                        cb_pi_true, cb_pi_infer):
  def compute_relative_freq(num_agents_lcl, num_latents_lcl, num_states_lcl,
                            trajs):
    list_rel_freq = [
        np.zeros((num_latents_lcl, num_states_lcl))
        for _ in range(num_agents_lcl)
    ]

    count = 0
    for traj in trajs:
      for s, joint_a, joint_x in traj:
        if joint_x[0] is None or joint_x[1] is None:
          continue
        for idx in range(num_agents_lcl):
          list_rel_freq[idx][joint_x[idx], s] += 1
        count += 1

    for idx in range(num_agents_lcl):
      list_rel_freq[idx] = list_rel_freq[idx] / count

    return list_rel_freq

  def compute_kl(agent_idx, x_idx, s_idx):
    np_p_inf = cb_pi_infer(agent_idx, x_idx, s_idx)
    np_p_true = cb_pi_true(agent_idx, x_idx, s_idx)

    sum_val = 0
    for idx in range(len(np_p_inf)):
      if np_p_inf[idx] != 0 and np_p_true[idx] != 0:
        sum_val += np_p_inf[idx] * (np.log(np_p_inf[idx]) -
                                    np.log(np_p_true[idx]))
    return sum_val

  list_rel_freq = compute_relative_freq(num_agents, num_latents, num_states,
                                        sax_trajectories)

  list_sum_kl = []
  for aidx in range(num_agents):
    sum_kl = 0
    for xidx in range(num_latents):
      for sidx in range(num_states):
        if list_rel_freq[aidx][xidx, sidx] > 0:
          sum_kl += list_rel_freq[aidx][xidx, sidx] * compute_kl(
              aidx, xidx, sidx)
    list_sum_kl.append(sum_kl)

  return tuple(list_sum_kl)