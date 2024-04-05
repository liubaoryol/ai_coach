import os
import glob
import numpy as np
import csv
from aic_domain.box_push_v2.simulator import BoxPushSimulatorV2
from aic_domain.rescue.simulator import RescueSimulator


def conv_survey_2_csv(survey_dir):
  header = [
      "ID", "Group", "Age", "Gender", "Freq", "PreComment", "A_Comment",
      "C_Comment", "PostComment", "PostQuestion"
  ]
  rows = []
  participants = glob.glob(os.path.join(survey_dir, "*"))
  for dir in participants:
    _, username = os.path.split(dir)
    user_row = [username, username[0]]
    prename = glob.glob(os.path.join(dir, "pre*"))[0]
    inname = glob.glob(os.path.join(dir, "in*"))[0]
    postname = glob.glob(os.path.join(dir, "post*"))[0]

    with open(prename, mode='r') as file:
      csvreader = csv.reader(file)
      pre_rows = [row for row in csvreader]

    with open(inname, mode='r') as file:
      csvreader = csv.reader(file)
      in_rows = [row for row in csvreader]

    with open(postname, mode='r') as file:
      csvreader = csv.reader(file)
      post_rows = [row for row in csvreader]

    user_row.extend(pre_rows[1][1:])
    user_row.append(in_rows[1][4])
    user_row.append(in_rows[2][4])
    user_row.extend(post_rows[1][2:])
    rows.append(user_row)

  cur_dir = os.path.dirname(__file__)
  save_path = os.path.join(cur_dir, "human_tic_comments.csv")
  with open(save_path, 'w', newline='') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerow(header)
    for row in rows:
      csv_writer.writerow(row)


def conv_data_2_csv(traj_dir):

  header = ["ID", "Group", "A1", "A2", "A3", "A4", "C1", "C2", "C3", "C4"]
  rows = []
  participants = glob.glob(os.path.join(traj_dir, "*"))
  TRAJ_PREFIX = "ntrv_session_"
  for dir in participants:
    _, username = os.path.split(dir)
    user_row = [username, username[0]]
    for idx in range(4):
      tname = glob.glob(
          os.path.join(dir, TRAJ_PREFIX + "a" + str(idx + 1) + "*"))[0]
      traj = BoxPushSimulatorV2.read_file(tname)
      score = 150 - len(traj)
      user_row.append(score)
    for idx in range(4):
      tname = glob.glob(
          os.path.join(dir, TRAJ_PREFIX + "c" + str(idx + 1) + "*"))[0]
      traj = RescueSimulator.read_file(tname)
      score = traj[-1][0]
      user_row.append(score)
    rows.append(user_row)
  cur_dir = os.path.dirname(__file__)
  save_path = os.path.join(cur_dir, "human_tic_scores.csv")
  with open(save_path, 'w', newline='') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerow(header)
    for row in rows:
      csv_writer.writerow(row)


def get_survey_results_by_group(survey_dir, group_name):
  SURVEY_KEYS = [
      "common_fluent", "common_contributed", "common_improved",
      "coach_engagement", "coach_intelligent", "coach_trust", "coach_effective",
      "coach_timely", "coach_contributed"
  ]

  movers_survey = {}
  rescue_survey = {}

  participants = glob.glob(os.path.join(survey_dir, group_name + "*"))
  for dir in participants:
    for sname in glob.glob(os.path.join(dir, "insurvey*")):
      with open(sname, mode='r') as file:
        dict_csv = csv.DictReader(file)
        count = 0
        for row in dict_csv:
          count += 1
          # movers
          if row['session'] == "ntrv_session_a4":
            for key in SURVEY_KEYS:
              if row[key] != "N/A":
                item = movers_survey.get(key, [])
                item.append(float(row[key]))
                movers_survey[key] = item
          if row["session"] == "ntrv_session_c4":
            for key in SURVEY_KEYS:
              if row[key] != "N/A":
                item = rescue_survey.get(key, [])
                item.append(float(row[key]))
                rescue_survey[key] = item
        assert count == 2, sname

  assert len(movers_survey[SURVEY_KEYS[0]]) == len(participants)
  assert len(rescue_survey[SURVEY_KEYS[0]]) == len(participants)

  return movers_survey, rescue_survey


def get_test_trajs_by_group(traj_dir, group_name):
  movers_trajs = []
  rescue_trajs = []
  participants = glob.glob(os.path.join(traj_dir, group_name + "*"))
  TRAJ_PREFIX = "ntrv_session_"
  for dir in participants:
    # movers trajs
    # add only test sessions
    for tname in glob.glob(os.path.join(dir, TRAJ_PREFIX + "a*")):
      if TRAJ_PREFIX + "a1" not in tname:
        traj = BoxPushSimulatorV2.read_file(tname)
        movers_trajs.append(traj)

    for tname in glob.glob(os.path.join(dir, TRAJ_PREFIX + "c*")):
      if TRAJ_PREFIX + "c1" not in tname:
        traj = RescueSimulator.read_file(tname)
        rescue_trajs.append(traj)

  assert len(movers_trajs) == len(participants) * 3
  assert len(rescue_trajs) == len(participants) * 3

  return movers_trajs, rescue_trajs


def get_num_interventions(intv_dir):
  movers_n_intv = []
  rescue_n_intv = []
  INTV_PREFIX = "interventions_ntrv_session_"
  participants = glob.glob(os.path.join(intv_dir, "*"))
  for dir in participants:
    # movers
    for iname in glob.glob(os.path.join(dir, INTV_PREFIX + "a*")):
      with open(iname, newline='') as txtfile:
        lines = txtfile.readlines()
        movers_n_intv.append(len(lines) - 1)
    # rescue
    for iname in glob.glob(os.path.join(dir, INTV_PREFIX + "c*")):
      with open(iname, newline='') as txtfile:
        lines = txtfile.readlines()
        rescue_n_intv.append(len(lines) - 1)

  return movers_n_intv, rescue_n_intv


def print_results(survey_dir, traj_dir, intv_dir):

  # get test-session trajectories by group and domain
  a_movers_trajs, a_rescue_trajs = get_test_trajs_by_group(traj_dir, "a")
  b_movers_trajs, b_rescue_trajs = get_test_trajs_by_group(traj_dir, "b")
  print("# Group A", len(a_movers_trajs) / 3)
  print("# Group B", len(b_movers_trajs) / 3)

  # get rewards of each group
  a_movers_scores = [150 - len(trj) for trj in a_movers_trajs]
  a_rescue_scores = [trj[-1][0] for trj in a_rescue_trajs]

  b_movers_scores = [150 - len(trj) for trj in b_movers_trajs]
  b_rescue_scores = [trj[-1][0] for trj in b_rescue_trajs]

  # mean, std
  a_m_mean, a_m_std = np.mean(a_movers_scores), np.std(a_movers_scores)
  b_m_mean, b_m_std = np.mean(b_movers_scores), np.std(b_movers_scores)

  a_r_mean, a_r_std = np.mean(a_rescue_scores), np.std(a_rescue_scores)
  b_r_mean, b_r_std = np.mean(b_rescue_scores), np.std(b_rescue_scores)

  print(f"Movers Group A: {a_m_mean} +- {a_m_std}")
  print(f"Movers Group B: {b_m_mean} +- {b_m_std}")
  print(f"Rescue Group A: {a_r_mean} +- {a_r_std}")
  print(f"Rescue Group B: {b_r_mean} +- {b_r_std}")

  # num interventions
  movers_n_intvs, rescue_n_intvs = get_num_interventions(intv_dir)
  m_ni_mean, m_ni_std = np.mean(movers_n_intvs), np.std(movers_n_intvs)
  r_ni_mean, r_ni_std = np.mean(rescue_n_intvs), np.std(rescue_n_intvs)
  print(f"Movers Interventions: {m_ni_mean} +- {m_ni_std}")
  print(f"Rescue Interventions: {r_ni_mean} +- {r_ni_std}")

  # survey results
  a_movers_survey, a_rescue_survey = get_survey_results_by_group(
      survey_dir, "a")
  b_movers_survey, b_rescue_survey = get_survey_results_by_group(
      survey_dir, "b")

  for key in a_movers_survey:
    a_m_mean, a_m_std = (np.mean(a_movers_survey[key]),
                         np.std(a_movers_survey[key]))
    b_m_mean, b_m_std = (np.mean(b_movers_survey[key]),
                         np.std(b_movers_survey[key]))
    print(f"Movers A {key}: {a_m_mean} +- {a_m_std}")
    print(f"Movers B {key}: {b_m_mean} +- {b_m_std}")

  for key in a_rescue_survey:
    a_r_mean, a_r_std = (np.mean(a_rescue_survey[key]),
                         np.std(a_rescue_survey[key]))
    b_r_mean, b_r_std = (np.mean(b_rescue_survey[key]),
                         np.std(b_rescue_survey[key]))
    print(f"Rescue A {key}: {a_r_mean} +- {a_r_std}")
    print(f"Rescue B {key}: {b_r_mean} +- {b_r_std}")

  for key in b_movers_survey:
    if key in a_movers_survey:
      continue

    b_m_mean, b_m_std = (np.mean(b_movers_survey[key]),
                         np.std(b_movers_survey[key]))
    print(f"Movers B {key}: {b_m_mean} +- {b_m_std}")

  for key in b_rescue_survey:
    if key in a_rescue_survey:
      continue

    b_r_mean, b_r_std = (np.mean(b_rescue_survey[key]),
                         np.std(b_rescue_survey[key]))
    print(f"Rescue B {key}: {b_r_mean} +- {b_r_std}")


if __name__ == "__main__":
  cur_dir = os.path.dirname(__file__)
  data_dir = os.path.join(cur_dir, "data")
  survey_dir = os.path.join(data_dir, "tw2020_survey")
  traj_dir = os.path.join(data_dir, "tw2020_trajectory")
  intv_dir = os.path.join(data_dir, "tw2020_user_label")

  print_results(survey_dir, traj_dir, intv_dir)
  # conv_data_2_csv(traj_dir)
  # conv_survey_2_csv(survey_dir)
