import os
import glob
import click
import logging
import random
import numpy as np
# from aic_ml.BTIL import BTIL
from aic_ml.BTIL.btil_decentral import BTIL_Decen
from aic_ml.BTIL.btil_svi import BTIL_SVI
import load_domain


# yapf: disable
@click.command()
@click.option("--domain", type=str, default="rescue_2", help="movers / cleanup_v3 / rescue_2 /rescue_3")  # noqa: E501
@click.option("--synthetic", type=bool, default=True, help="")
@click.option("--num-training-data", type=int, default=160, help="")
@click.option("--supervision", type=float, default=0.3, help="value should be between 0.0 and 1.0")  # noqa: E501
@click.option("--gen-trainset", type=bool, default=False, help="")
@click.option("--beta-pi", type=float, default=0.01, help="")
@click.option("--beta-tx", type=float, default=0.01, help="")
@click.option("--tx-dependency", type=str, default="FTTT",
              help="sequence of T or F indicating dependency on cur_state, actions, and next_state")  # noqa: E501
@click.option("--batch-size", type=int, default=-1, help="If minus, use Decen. Otherwise, use SVI")  # noqa: E501
# yapf: enable
def main(domain, synthetic, num_training_data, supervision, gen_trainset,
         beta_pi, beta_tx, tx_dependency, batch_size):
  logging.info("domain: %s" % (domain, ))
  logging.info("synthetic: %s" % (synthetic, ))
  logging.info("num training data: %s" % (num_training_data, ))
  logging.info("supervision: %s" % (supervision, ))
  logging.info("Gen trainset: %s" % (gen_trainset, ))
  logging.info("beta pi: %s" % (beta_pi, ))
  logging.info("beta Tx: %s" % (beta_tx, ))
  logging.info("Tx dependency: %s" % (tx_dependency, ))
  logging.info("batch size: %s" % (batch_size, ))

  # define the domain where trajectories were generated
  ##################################################
  if domain == "movers":
    vec_domain_data = load_domain.load_movers()
  elif domain == "cleanup_v2":
    vec_domain_data = load_domain.load_cleanup_v2()
  elif domain == "cleanup_v3":
    vec_domain_data = load_domain.load_cleanup_v3()
  elif domain == "rescue_2":
    vec_domain_data = load_domain.load_rescue_2()
  elif domain == "rescue_3":
    vec_domain_data = load_domain.load_rescue_3()
  else:
    raise NotImplementedError

  sim, AGENTS, SAVE_PREFIX, train_data, GAME_MAP = vec_domain_data
  sim.init_game(**GAME_MAP)
  sim.set_autonomous_agent(*AGENTS)

  num_states = AGENTS[0].agent_model.get_reference_mdp().num_states
  tuple_num_actions = []
  tuple_num_latents = []
  for agent in AGENTS:
    tuple_num_actions.append(agent.agent_model.policy_model.get_num_actions())
    tuple_num_latents.append(
        agent.agent_model.policy_model.get_num_latent_states())
  tuple_num_actions = tuple(tuple_num_actions)
  tuple_num_latents = tuple(tuple_num_latents)

  tuple_tx_dependency = []
  for cha in tx_dependency:
    if cha == "T":
      tuple_tx_dependency.append(True)
    else:
      tuple_tx_dependency.append(False)

  tuple_tx_dependency = tuple(tuple_tx_dependency)

  # generate data
  ############################################################################
  if synthetic:
    dir_name = "data/"
  else:
    dir_name = "human_data/"

  DATA_DIR = os.path.join(os.path.dirname(__file__), dir_name)
  TRAIN_DIR = os.path.join(DATA_DIR, SAVE_PREFIX + '_train')

  train_prefix = "train_"
  if gen_trainset:
    file_names = glob.glob(os.path.join(TRAIN_DIR, train_prefix + '*.txt'))
    for fmn in file_names:
      os.remove(fmn)
    sim.run_simulation(num_training_data, os.path.join(TRAIN_DIR, train_prefix),
                       "header")

  fn_get_bx = None
  if synthetic:
    fn_get_bx = (
        lambda a, s: np.ones(tuple_num_latents[a]) / tuple_num_latents[a])
  else:
    fn_get_bx = (
        lambda a, s: np.ones(tuple_num_latents[a]) / tuple_num_latents[a])

  # load train set
  ##################################################
  file_names = glob.glob(os.path.join(TRAIN_DIR, '*.txt'))
  random.shuffle(file_names)

  num_train = min(num_training_data, len(file_names))
  logging.info(num_train)

  train_files = file_names[:num_train]

  train_data.load_from_files(train_files)
  traj_labeled_ver = train_data.get_as_row_lists(no_latent_label=False,
                                                 include_terminal=False)
  traj_unlabel_ver = train_data.get_as_row_lists(no_latent_label=True,
                                                 include_terminal=False)

  logging.info(len(traj_labeled_ver))

  # learn policy and transition
  ##################################################
  logging.info("beta: %f, %f" % (beta_pi, beta_tx))

  labeled_data_idx = int(num_train * supervision)

  logging.info("#########")
  logging.info("BTIL (Labeled: %d, Unlabeled: %d)" %
               (labeled_data_idx, num_train - labeled_data_idx))
  logging.info("#########")

  # save models
  save_dir = DATA_DIR + "learned_models/"
  if not os.path.exists(save_dir):
    os.mkdir(save_dir)

  # learning models
  if batch_size < 0:
    alg_name = "btil_dec"
    btil_models = BTIL_Decen(traj_labeled_ver[0:labeled_data_idx] +
                             traj_unlabel_ver[labeled_data_idx:],
                             num_states,
                             tuple_num_latents,
                             tuple_num_actions,
                             trans_x_dependency=tuple_tx_dependency,
                             epsilon=0.01,
                             max_iteration=100)
    btil_models.set_dirichlet_prior(beta_pi, beta_tx)

    btil_models.set_bx_and_Tx(cb_bx=fn_get_bx)
    btil_models.do_inference()
  else:
    alg_name = "btil_svi"
    btil_models = BTIL_SVI(traj_labeled_ver[0:labeled_data_idx] +
                           traj_unlabel_ver[labeled_data_idx:],
                           num_states,
                           tuple_num_latents,
                           tuple_num_actions,
                           trans_x_dependency=tuple_tx_dependency,
                           epsilon=0.01,
                           max_iteration=300,
                           lr=0.1,
                           decay=0,
                           no_gem=True)
    btil_models.set_prior(None, beta_pi, beta_tx)

    btil_models.initialize_param()
    btil_models.do_inference(batch_size)

    # save initial latent distribution
    bx_file_name = SAVE_PREFIX + f"_{alg_name}_bx_"
    bx_file_name += "synth_" if synthetic else "human_"
    bx_file_name += tx_dependency + "_"
    bx_file_name += ("%d_%.2f" % (num_train, supervision)).replace('.', ',')
    bx_file_name = os.path.join(save_dir, bx_file_name)
    for idx in range(len(btil_models.list_bx)):
      np.save(bx_file_name + f"_a{idx + 1}", btil_models.list_bx[idx])

  policy_file_name = SAVE_PREFIX + f"_{alg_name}_policy_"
  policy_file_name += "synth_" if synthetic else "human_"
  policy_file_name += tx_dependency + "_"
  policy_file_name += ("%d_%.2f" % (num_train, supervision)).replace('.', ',')
  policy_file_name = os.path.join(save_dir, policy_file_name)
  for idx in range(len(btil_models.list_np_policy)):
    np.save(policy_file_name + f"_a{idx + 1}", btil_models.list_np_policy[idx])

  tx_file_name = SAVE_PREFIX + f"_{alg_name}_tx_"
  tx_file_name += "synth_" if synthetic else "human_"
  tx_file_name += tx_dependency + "_"
  tx_file_name += ("%d_%.2f" % (num_train, supervision)).replace('.', ',')
  tx_file_name = os.path.join(save_dir, tx_file_name)
  for idx in range(len(btil_models.list_Tx)):
    np.save(tx_file_name + f"_a{idx + 1}", btil_models.list_Tx[idx].np_Tx)


if __name__ == "__main__":
  logging.basicConfig(
      level=logging.INFO,
      format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
      handlers=[logging.StreamHandler()],
      force=True)
  main()
