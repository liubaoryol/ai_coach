import numpy as np
import os
import pickle
import logging
import click
from tqdm import tqdm
from ai_coach_core.intervention.full_mdp import FullMDP
from ai_coach_core.models.mdp import v_value_from_policy

from ai_coach_domain.rescue_v2 import E_EventType
from ai_coach_domain.rescue_v2 import is_work_done
from ai_coach_domain.rescue_v2.transition import find_location_index
from ai_coach_domain.rescue_v2.maps import MAP_RESCUE
from ai_coach_domain.rescue_v2.policy import Policy_Rescue
from ai_coach_domain.rescue_v2.mdp import MDP_Rescue_Agent
from ai_coach_domain.rescue_v2.mdp import MDP_Rescue_Task
from ai_coach_domain.rescue_v2.agent import RescueAM


class FullMDP_Rescue2(FullMDP):

  def reward(self, state_idx: int, action_idx: int) -> float:
    if self.is_terminal(state_idx):
      return 0

    self.mmdp = self.mmdp  # type: MDP_Rescue_Task

    state_vec = self.conv_idx_to_state(state_idx)
    obs_vec = state_vec[:self.mmdp.num_state_factors]
    obs_idx = self.mmdp.conv_state_to_idx(tuple(obs_vec))

    work_states, pos1, pos2, pos3 = self.mmdp.conv_mdp_sidx_to_sim_states(
        obs_idx)
    act1, act2, act3 = self.mmdp.conv_mdp_aidx_to_sim_actions(action_idx)

    reward = 0
    for idx in range(len(work_states)):
      # if work_states[idx] != 0:
      if not is_work_done(idx, work_states,
                          self.mmdp.work_info[idx].coupled_works):
        workload = self.mmdp.work_info[idx].workload

        work1 = find_location_index(self.mmdp.work_locations, pos1)
        work2 = find_location_index(self.mmdp.work_locations, pos2)
        work3 = find_location_index(self.mmdp.work_locations, pos3)
        workforce = 0
        if work1 == idx and act1 == E_EventType.Rescue:
          workforce += 1
        if work2 == idx and act2 == E_EventType.Rescue:
          workforce += 1
        if work3 == idx and act3 == E_EventType.Rescue:
          workforce += 1

        if workload <= workforce:
          place_id = self.mmdp.work_info[idx].rescue_place
          reward += self.mmdp.places[place_id].helps
    # if reward > 0:
    #   print(reward)
    #   print(work_states, pos1, pos2)
    #   print(act1, act2)

    return reward


# yapf: disable
@click.command()
@click.option("--domain", type=str, default="rescue_3", help="rescue_3")
@click.option("--iteration", type=int, default=15, help="")
@click.option("--num-train", type=int, default=500, help="")
@click.option("--supervision", type=float, default=0.3, help="value should be between 0.0 and 1.0")  # noqa: E501
# yapf: enable
def save_merged_v_values(domain, iteration, num_train=500, supervision=0.3):
  TRUE_MODELS = False
  # domain: movers | cleanup_v2 | rescue_2

  sup_txt = ("%.2f" % supervision).replace('.', ',')
  policy1_file = domain + f"_btil2_policy_synth_woTx_FTTTT_{num_train}_{sup_txt}_a1.npy"
  policy2_file = domain + f"_btil2_policy_synth_woTx_FTTTT_{num_train}_{sup_txt}_a2.npy"
  policy3_file = domain + f"_btil2_policy_synth_woTx_FTTTT_{num_train}_{sup_txt}_a3.npy"
  tx1_file = domain + f"_btil2_tx_synth_FTTTT_{num_train}_{sup_txt}_a1.npy"
  tx2_file = domain + f"_btil2_tx_synth_FTTTT_{num_train}_{sup_txt}_a2.npy"
  tx3_file = domain + f"_btil2_tx_synth_FTTTT_{num_train}_{sup_txt}_a3.npy"

  if domain == "rescue_3":
    task_mdp = MDP_Rescue_Task(**MAP_RESCUE)
    agent_mdp = MDP_Rescue_Agent(**MAP_RESCUE)
    tup_lstate = (agent_mdp.latent_space, agent_mdp.latent_space,
                  agent_mdp.latent_space)

    TEMPERATURE = 0.3
    # true models
    policy_a1 = Policy_Rescue(task_mdp, agent_mdp, TEMPERATURE, 0)
    policy_a2 = Policy_Rescue(task_mdp, agent_mdp, TEMPERATURE, 1)
    policy_a3 = Policy_Rescue(task_mdp, agent_mdp, TEMPERATURE, 2)
    agent_model_a1 = RescueAM(0, policy_a1)
    agent_model_a2 = RescueAM(1, policy_a2)
    agent_model_a3 = RescueAM(2, policy_a3)
    agent_models = [agent_model_a1, agent_model_a2, agent_model_a3]

    FullMDP_Base = FullMDP_Rescue2

  DATA_DIR = os.path.join(os.path.dirname(__file__), "data/")
  model_dir = os.path.join(DATA_DIR, "learned_models/")

  if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

  if TRUE_MODELS:
    tup_num_latent = (policy_a1.get_num_latent_states(),
                      policy_a2.get_num_latent_states(),
                      policy_a3.get_num_latent_states())

    def all_Tx(agent_idx, latstate_idx, obstate_idx, tuple_action_idx,
               obstate_next_idx):
      agent_models[agent_idx].transition_mental_state(latstate_idx, obstate_idx,
                                                      tuple_action_idx,
                                                      obstate_next_idx)
  else:
    np_tx1 = np.load(model_dir + tx1_file)
    np_tx2 = np.load(model_dir + tx2_file)
    np_tx3 = np.load(model_dir + tx3_file)

    np_policy1 = np.load(model_dir + policy1_file)
    np_policy2 = np.load(model_dir + policy2_file)
    np_policy3 = np.load(model_dir + policy3_file)

    tup_num_latent = (np_policy1.shape[0], np_policy2.shape[0],
                      np_policy3.shape[0])

    def all_Tx(agent_idx, latstate_idx, obstate_idx, tuple_action_idx,
               obstate_next_idx):
      np_dist = None
      np_idx = tuple([latstate_idx, *tuple_action_idx, obstate_next_idx])
      if agent_idx == 0:
        np_dist = np_tx1[np_idx]
      elif agent_idx == 1:
        np_dist = np_tx2[np_idx]
      elif agent_idx == 2:
        np_dist = np_tx3[np_idx]
      else:
        raise NotImplementedError

      # for illegal states or states that haven't appeared during the training,
      # we assume mental model was maintained.
      if np.all(np_dist == np_dist[0]):
        np_dist = np.zeros_like(np_dist)
        np_dist[latstate_idx] = 1

      return np_dist

  full_mdp = FullMDP_Base(mmdp=task_mdp, cb_tx=all_Tx, tup_lstate=tup_lstate)
  # full mdp transition
  full_transition_file_name = domain + f"_{num_train}_{sup_txt}" + "_full_transition"
  full_transition_file_name += "" if TRUE_MODELS else "_learned"
  pickle_full_trans = os.path.join(DATA_DIR,
                                   full_transition_file_name + ".pickle")

  if os.path.exists(pickle_full_trans):
    with open(pickle_full_trans, 'rb') as handle:
      np_transition_model = pickle.load(handle)
  else:
    np_transition_model = full_mdp.np_transition_model
    with open(pickle_full_trans, 'wb') as handle:
      pickle.dump(np_transition_model, handle, protocol=pickle.HIGHEST_PROTOCOL)

  # full mdp reward
  reward_file_name = domain + "_reward"
  pickle_reward = os.path.join(DATA_DIR, reward_file_name + ".pickle")

  if os.path.exists(pickle_reward):
    with open(pickle_reward, 'rb') as handle:
      np_reward_model = pickle.load(handle)
  else:
    np_reward_model = full_mdp.np_reward_model
    with open(pickle_reward, 'wb') as handle:
      pickle.dump(np_reward_model, handle, protocol=pickle.HIGHEST_PROTOCOL)

  # joint policy
  policy_file_name = domain + f"_{num_train}_{sup_txt}" + "_joint_policy"
  policy_file_name += "" if TRUE_MODELS else "_learned"
  pickle_joint_policy = os.path.join(DATA_DIR, policy_file_name + ".pickle")

  if os.path.exists(pickle_joint_policy):
    with open(pickle_joint_policy, 'rb') as handle:
      np_policy = pickle.load(handle)
  else:
    np_policy = np.zeros((full_mdp.num_states, full_mdp.num_actions))
    if TRUE_MODELS:
      raise NotImplementedError
    else:
      stay_aidx = full_mdp.conv_sim_actions_to_mdp_aidx(
          (E_EventType.Stay, E_EventType.Stay, E_EventType.Stay))
      for obs_idx in tqdm(range(task_mdp.num_states)):
        obs_vec = task_mdp.conv_idx_to_state(obs_idx)

        for xidx1 in range(tup_num_latent[0]):
          np_action1 = np_policy1[xidx1, obs_idx]
          for xidx2 in range(tup_num_latent[1]):
            np_action2 = np_policy2[xidx2, obs_idx]
            for xidx3 in range(tup_num_latent[2]):
              np_action3 = np_policy3[xidx3, obs_idx]

              state_vec = tuple(obs_vec) + (xidx1, xidx2, xidx3)
              sidx = full_mdp.conv_state_to_idx(state_vec)

              legal_actions = full_mdp.legal_actions(sidx)
              if len(legal_actions) == 0:
                np_policy[sidx, stay_aidx] = 1
              else:
                for aidx in legal_actions:
                  act1, act2, act3 = full_mdp.conv_idx_to_action(aidx)
                  np_policy[sidx, aidx] = (np_action1[act1] * np_action2[act2] *
                                           np_action3[act3])
      np_policy = np_policy / np.sum(np_policy, axis=1)[:, np.newaxis]

    with open(pickle_joint_policy, 'wb') as handle:
      pickle.dump(np_policy, handle, protocol=pickle.HIGHEST_PROTOCOL)

  # v-value
  file_name = domain + f"_{num_train}_{sup_txt}_{iteration}" + "_merged_v_values"
  file_name += "" if TRUE_MODELS else "_learned"
  pickle_v_values = os.path.join(DATA_DIR, file_name + ".pickle")
  if os.path.exists(pickle_v_values):
    with open(pickle_v_values, 'rb') as handle:
      np_v_values = pickle.load(handle)
  else:
    np_v_values = v_value_from_policy(np_policy,
                                      np_transition_model,
                                      np_reward_model,
                                      max_iteration=iteration,
                                      epsilon=0.1,
                                      discount_factor=1.)
    np_v_values.shape = (task_mdp.num_states, tup_num_latent[0],
                         tup_num_latent[1])

    logging.info("save v_value by pickle")
    with open(pickle_v_values, 'wb') as handle:
      pickle.dump(np_v_values, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
  save_merged_v_values()
