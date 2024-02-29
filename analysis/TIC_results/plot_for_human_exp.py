import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="white")


def save_box_plots(df,
                   output_name,
                   domain,
                   domain_name,
                   perfect_score,
                   perfect_step,
                   save_plot,
                   cost=1):

  df["real_score"] = df["score"] - df["num_feedback"] * df["cost"]

  handle = None
  fig, axes = plt.subplots(1, 2, figsize=(10, 5))

  df_domain = df[df["domain"] == domain]
  # ==== reward vs delta =====
  reward_vs_delta = df_domain[((df_domain["strategy"] == "Average")
                               | (df_domain["strategy"] == "Argmax")
                               | (df_domain["strategy"] == "Argmax_robot_fix"))
                              & (df_domain["cost"] == cost)]
  assert len(reward_vs_delta) == 2700

  # all_interv_arg = df_domain[(df_domain["strategy"] == "Argmax")
  #                            & (df_domain["interv_thres"] == 0)]
  no_interv = df_domain[(df_domain["strategy"] == "No_intervention")
                        & (df_domain["cost"] == cost)]
  assert len(no_interv) == 100
  # all_interv_arg_score = all_interv_arg["score"].mean()
  no_interv_score = no_interv["score"].mean()

  line_width = 1.5
  ax = sns.lineplot(ax=axes[0],
                    data=reward_vs_delta,
                    x='interv_thres',
                    y='score',
                    hue="strategy",
                    lw=line_width)
  if domain == "movers":
    rule_based = df_domain[(df_domain["strategy"] == "Rule_avg")
                           & (df_domain["cost"] == cost)]
    assert len(rule_based) == 100
    rule_based_score = rule_based["score"].mean()
    ax.axhline(rule_based_score,
               label="Expectation-Rule",
               color='c',
               lw=line_width)

  perfect_scr = perfect_score
  ax.axhline(perfect_scr,
             label="Centralized policy",
             color='green',
             ls='-.',
             lw=line_width)
  ax.axhline(no_interv_score,
             label="No intervention",
             color='black',
             ls='--',
             lw=line_width)

  h, labels = ax.get_legend_handles_labels()
  labels[labels.index("Average")] = "Expectation-Value"
  labels[labels.index("Argmax")] = "Deterministic-Value"
  labels[labels.index("Argmax_robot_fix")] = "Determ-Value-FixRobot"

  fontsize = 16
  ax.set_ylabel("Task reward", fontsize=fontsize)
  ax.set_xlabel(r"Benefit threshold ($\delta$)")
  ax.set_xlabel(r"Benefit threshold ($\delta$)", fontsize=fontsize)
  ax.set_title(domain_name, fontsize=fontsize + 2)
  handle = h
  ax.legend([], [], frameon=False)

  ax = sns.lineplot(ax=axes[1],
                    data=reward_vs_delta,
                    x='interv_thres',
                    y='real_score',
                    hue="strategy",
                    lw=line_width)
  if domain == "movers":
    rule_based = df_domain[(df_domain["strategy"] == "Rule_avg")
                           & (df_domain["cost"] == cost)]
    assert len(rule_based) == 100
    rule_based_score = rule_based["real_score"].mean()
    ax.axhline(rule_based_score,
               label="Expectation-Rule",
               color='c',
               lw=line_width)

  perfect_scr = perfect_score - 1 * perfect_step
  ax.axhline(perfect_scr,
             label="Centralized policy",
             color='green',
             ls='-.',
             lw=line_width)
  ax.axhline(no_interv_score,
             label="No intervention",
             color='black',
             ls='--',
             lw=line_width)

  ax.set_ylabel("Objective(J)", fontsize=fontsize)
  ax.set_xlabel(r"Benefit threshold ($\delta$)")
  ax.set_xlabel(r"Benefit threshold ($\delta$)", fontsize=fontsize)
  ax.set_title(domain_name, fontsize=fontsize + 2)
  ax.legend([], [], frameon=False)
  fig.legend(handle,
             labels,
             loc='lower center',
             ncol=len(labels),
             bbox_to_anchor=(0.5, 0.0),
             columnspacing=0.8,
             prop={'size': 10})
  fig.tight_layout()
  fig.subplots_adjust(bottom=0.2)
  if save_plot:
    fig.savefig(output_name)


def save_rescue_plots(df,
                      output_name,
                      domain,
                      domain_name,
                      perfect_score,
                      perfect_step,
                      save_plot,
                      cost=1):

  df["real_score"] = df["score"] - df["num_feedback"] * df["cost"]

  handle = None
  fig, axes = plt.subplots(1, 2, figsize=(10, 5))
  df_domain = df[df["domain"] == domain]
  # ==== reward vs delta =====
  reward_vs_delta = df_domain[((df_domain["strategy"] == "Average")
                               | (df_domain["strategy"] == "Argmax")
                               | (df_domain["strategy"] == "Argmax_robot_fix"))
                              & (df_domain["cost"] == cost)]
  assert len(reward_vs_delta) == 2700

  # all_interv_arg = df_domain[(df_domain["strategy"] == "Argmax")
  #                            & (df_domain["interv_thres"] == 0)]
  no_interv = df_domain[(df_domain["strategy"] == "No_intervention")
                        & (df_domain["cost"] == cost)]
  assert len(no_interv) == 100
  # all_interv_arg_score = all_interv_arg["score"].mean()
  no_interv_score = no_interv["score"].mean()

  line_width = 1.5
  ax = sns.lineplot(ax=axes[0],
                    data=reward_vs_delta,
                    x='interv_thres',
                    y='score',
                    hue="strategy",
                    lw=line_width)

  perfect_scr = perfect_score
  ax.axhline(perfect_scr,
             label="Centralized policy",
             color='green',
             ls='-.',
             lw=line_width)
  ax.axhline(no_interv_score,
             label="No intervention",
             color='black',
             ls='--',
             lw=line_width)

  h, labels = ax.get_legend_handles_labels()
  labels[labels.index("Average")] = "Expectation-Value"
  labels[labels.index("Argmax")] = "Deterministic-Value"
  labels[labels.index("Argmax_robot_fix")] = "Determ-Value-FixRobot"
  fontsize = 16
  ax.set_ylabel("Task reward", fontsize=fontsize)
  ax.set_xlabel(r"Benefit threshold ($\delta$)")
  ax.set_xlabel(r"Benefit threshold ($\delta$)", fontsize=fontsize)
  ax.set_title(domain_name, fontsize=fontsize + 2)
  if handle is None:
    handle = h
  ax.legend([], [], frameon=False)

  ax = sns.lineplot(ax=axes[1],
                    data=reward_vs_delta,
                    x='interv_thres',
                    y='num_feedback',
                    hue="strategy",
                    lw=line_width)

  ax.set_ylabel("# intervention", fontsize=fontsize)
  ax.set_title(domain_name, fontsize=fontsize + 2)
  ax.set_xlabel(r"Benefit threshold ($\delta$)", fontsize=fontsize)
  ax.legend([], [], frameon=False)

  fig.legend(handle,
             labels,
             loc='lower center',
             ncol=len(labels),
             columnspacing=1.3,
             bbox_to_anchor=(0.5, 0.0),
             prop={'size': 10})
  fig.tight_layout()
  fig.subplots_adjust(bottom=0.2)
  if save_plot:
    fig.savefig(output_name)


def save_score_vs_theta_plots(df,
                              output_name,
                              domain,
                              domain_name,
                              perfect_score,
                              perfect_step,
                              cost,
                              interv_thres,
                              save_plot,
                              consider_cost=True):
  df["real_score"] = df["score"] - df["num_feedback"] * df["cost"]
  metric = "score" if not consider_cost else "real_score"
  metric_ylabel = "Reward" if not consider_cost else "Objective(J)"

  num_domain = 1

  fig, axes = plt.subplots(1, num_domain, figsize=(5 * num_domain, 4))
  df_domain = df[df["domain"] == domain]
  score_vs_theta = df_domain[(
      ((df_domain["strategy"] == "Argmax_thres") &
       (df_domain["interv_thres"] == interv_thres))
      | ((df_domain["strategy"] == "Argmax_thres_robot_fix") &
         (df_domain["interv_thres"] == interv_thres))
      | (df_domain["strategy"] == "Rule_thres"))
                             & (df_domain["cost"] == cost)]

  no_interv = df_domain[(df_domain["strategy"] == "No_intervention")
                        & (df_domain["cost"] == cost)]

  avg_rule = df_domain[(df_domain["strategy"] == "Rule_avg")
                       & (df_domain["cost"] == cost)]

  avg_value = df_domain[(df_domain["strategy"] == "Average")
                        & (df_domain["interv_thres"] == interv_thres)
                        & (df_domain["cost"] == cost)]

  ax = sns.lineplot(ax=axes,
                    data=score_vs_theta,
                    x='infer_thres',
                    y=metric,
                    hue="strategy")
  line_width = 2
  ax.axhline(avg_value[metric].mean(),
             label=r"Expectation-Value($\delta=$" + f"{interv_thres})",
             color='m',
             ls=':',
             lw=line_width)

  if len(avg_rule[metric]) > 0:
    ax.axhline(avg_rule[metric].mean(),
               label="Expectation-Rule",
               color='c',
               ls='-',
               lw=line_width)
  central_score = (perfect_score if not consider_cost else perfect_score -
                   cost * perfect_step)
  ax.axhline(central_score,
             label="Centralized policy",
             color='green',
             ls='-.',
             lw=line_width)
  ax.axhline(no_interv[metric].mean(),
             label="No intervention",
             color='black',
             ls='--',
             lw=line_width)

  h, labels = ax.get_legend_handles_labels()
  labels[labels.index("Argmax_thres")] = (r"Confidence-Value($\delta=$" +
                                          f"{interv_thres})")
  labels[labels.index("Argmax_thres_robot_fix")] = (
      r"Conf-Value-FixRobot($\delta=$" + f"{interv_thres})")
  if "Rule_thres" in labels:
    labels[labels.index("Rule_thres")] = "Confidence-Rule"

  fontsize = 13
  ax.set_ylabel(metric_ylabel, fontsize=fontsize)
  ax.set_xlabel(r"Inference threshold ($\theta$)", fontsize=fontsize)
  ax.set_title(domain_name, fontsize=fontsize + 2)
  ax.legend(h, labels, prop={'size': 8})
  fig.tight_layout()
  if save_plot:
    fig.savefig(output_name)


def save_score_vs_intervention_plots(df,
                                     output_name,
                                     domain,
                                     domain_name,
                                     perfect_score,
                                     perfect_step,
                                     cost,
                                     save_plot,
                                     consider_cost=True,
                                     plot_center_policy=False):
  df["real_score"] = df["score"] - df["num_feedback"] * df["cost"]
  metric = "score" if not consider_cost else "real_score"
  metric_ylabel = "Reward" if not consider_cost else "Objective(J)"

  print(len(df))
  df = df[(df.strategy != "Argmax_thres_robot_fix")
          & (df.strategy != "Argmax_robot_fix")
          & (df.cost == cost)]
  print(len(df))

  num_domain = 1

  fig, axes = plt.subplots(1, num_domain, figsize=(5 * num_domain, 4))
  df_domain = df[df["domain"] == domain]
  print(len(df_domain))
  df_domain = df_domain.replace(to_replace="Argmax",
                                value='Det-Val',
                                regex=False)
  df_domain = df_domain.replace(to_replace="Argmax_thres",
                                value='Con-Val',
                                regex=False)
  df_domain = df_domain.replace(to_replace="Average",
                                value='Avg-Val',
                                regex=False)
  df_domain = df_domain.replace(to_replace="Rule_thres",
                                value='Con-Rul',
                                regex=False)
  df_domain = df_domain.replace(to_replace="Rule_avg",
                                value='Avg-Rul',
                                regex=False)
  df_domain = df_domain.replace(to_replace="No_intervention",
                                value='No-Int',
                                regex=False)

  df_mean = df_domain.groupby(['strategy', 'interv_thres',
                               'infer_thres'])[[metric, "num_feedback"]].mean()
  # print(df_mean)

  ax = sns.scatterplot(ax=axes,
                       data=df_mean,
                       x='num_feedback',
                       y=metric,
                       hue='interv_thres',
                       style='strategy',
                       s=70)

  if plot_center_policy:
    perfect_scr = (perfect_score if not consider_cost else perfect_score -
                   cost * perfect_step)
    ax.plot(perfect_step, perfect_scr, 'c^', label="Centralized policy")

  fontsize = 13
  ax.set_xlabel("# intervention", fontsize=fontsize)
  ax.set_ylabel(metric_ylabel, fontsize=fontsize)
  ax.set_title(domain_name, fontsize=fontsize + 2)
  for index, row in df_mean.iterrows():
    point_label = index[0] + f"-{index[1]:.1f}-{index[2]:.1f}"
    ax.annotate(point_label, (row['num_feedback'], row[metric]), fontsize=8)

  h, labels = ax.get_legend_handles_labels()
  # ax.legend(h, labels, prop={'size': 8})
  ax.legend([], [], frameon=False)

  fig.legend(h,
             labels,
             loc='lower center',
             ncol=int((len(labels) + 1) / 2),
             bbox_to_anchor=(0.5, 0.0),
             columnspacing=0.8,
             prop={'size': 8})
  fig.tight_layout()
  fig.subplots_adjust(bottom=0.2)

  if save_plot:
    fig.savefig(output_name)


if __name__ == "__main__":
  cur_dir = os.path.dirname(__file__)
  result_dir = os.path.join(cur_dir, "human_data/intervention_results/")
  output_dir = os.path.join(cur_dir, "human_output/")

  MOVERS = "movers"
  CLEANUP = "cleanup_v3"
  FLOOD = "rescue_2"
  BLACKOUT = "rescue_3"

  list_domains = [MOVERS, FLOOD]

  intv_result_name = "intv_res_20240216"
  list_intv_files = [
      os.path.join(result_dir, intv_result_name + f"-{dname}.csv")
      for dname in list_domains
  ]

  prefix = ""

  MAP_DOMAIN_NAME = {
      MOVERS: "Movers",
      CLEANUP: "Cleanup",
      FLOOD: "Flood",
      BLACKOUT: "Blackout"
  }

  MAP_SCORE = {MOVERS: -43, CLEANUP: -21, FLOOD: 7, BLACKOUT: 5}

  MAP_STEP = {MOVERS: 43, CLEANUP: 21, FLOOD: 19, BLACKOUT: 9}

  DICT_INTERV_THRES = {
      MOVERS: [0, 1, 3, 5, 10, 15, 20, 30, 50],
      CLEANUP: [0, 0.3, 0.5, 1, 2, 5, 10, 15, 20],
      FLOOD: [0, 0.1, 0.3, 0.5, 1, 1.5, 2, 3, 5],
      BLACKOUT: [0, 0.1, 0.2, 0.3, 0.5, 1, 1.5, 2, 3],
  }

  movers_cost = 1
  rescue_cost = 0

  SAVE_RESULT = True
  NO_SAVE = not SAVE_RESULT

  df_intv_res = pd.concat(map(pd.read_csv, list_intv_files), ignore_index=True)

  PLOT_MOVERS = False
  PLOT_RESCUE = True

  TEST_PLOT = False
  if TEST_PLOT:
    save_score_vs_intervention_plots(df_intv_res,
                                     output_dir + prefix +
                                     "Score_NumInt_rescue.png",
                                     FLOOD,
                                     MAP_DOMAIN_NAME[FLOOD],
                                     MAP_SCORE[FLOOD],
                                     MAP_STEP[FLOOD],
                                     rescue_cost,
                                     save_plot=SAVE_RESULT,
                                     consider_cost=False,
                                     plot_center_policy=True)
    plt.show()
    raise NotImplementedError("Test plot")

  if PLOT_MOVERS:
    save_box_plots(df_intv_res,
                   output_dir + prefix + "delta_movers.png",
                   MOVERS,
                   MAP_DOMAIN_NAME[MOVERS],
                   MAP_SCORE[MOVERS],
                   MAP_STEP[MOVERS],
                   SAVE_RESULT,
                   cost=movers_cost)

    save_score_vs_theta_plots(df_intv_res,
                              output_dir + prefix + "theta_movers.png",
                              MOVERS,
                              MAP_DOMAIN_NAME[MOVERS],
                              MAP_SCORE[MOVERS],
                              MAP_STEP[MOVERS],
                              cost=movers_cost,
                              interv_thres=5,
                              save_plot=SAVE_RESULT,
                              consider_cost=True)

    save_score_vs_theta_plots(df_intv_res,
                              output_dir + prefix + "theta_movers.png",
                              MOVERS,
                              MAP_DOMAIN_NAME[MOVERS],
                              MAP_SCORE[MOVERS],
                              MAP_STEP[MOVERS],
                              cost=movers_cost,
                              interv_thres=3,
                              save_plot=SAVE_RESULT,
                              consider_cost=True)

    save_score_vs_intervention_plots(df_intv_res,
                                     output_dir + prefix +
                                     "Score_NumInt_movers.png",
                                     MOVERS,
                                     MAP_DOMAIN_NAME[MOVERS],
                                     MAP_SCORE[MOVERS],
                                     MAP_STEP[MOVERS],
                                     movers_cost,
                                     save_plot=SAVE_RESULT,
                                     consider_cost=True)

  if PLOT_RESCUE:
    save_rescue_plots(df_intv_res,
                      output_dir + prefix + "delta_rescue.png",
                      FLOOD,
                      MAP_DOMAIN_NAME[FLOOD],
                      MAP_SCORE[FLOOD],
                      MAP_STEP[FLOOD],
                      SAVE_RESULT,
                      cost=rescue_cost)

    save_score_vs_theta_plots(df_intv_res,
                              output_dir + prefix + "theta_rescue.png",
                              FLOOD,
                              MAP_DOMAIN_NAME[FLOOD],
                              MAP_SCORE[FLOOD],
                              MAP_STEP[FLOOD],
                              cost=rescue_cost,
                              interv_thres=0,
                              save_plot=SAVE_RESULT,
                              consider_cost=False)

    save_score_vs_theta_plots(df_intv_res,
                              output_dir + prefix + "theta_rescue.png",
                              FLOOD,
                              MAP_DOMAIN_NAME[FLOOD],
                              MAP_SCORE[FLOOD],
                              MAP_STEP[FLOOD],
                              cost=rescue_cost,
                              interv_thres=0.5,
                              save_plot=SAVE_RESULT,
                              consider_cost=False)

    save_score_vs_intervention_plots(df_intv_res,
                                     output_dir + prefix +
                                     "Score_NumInt_rescue.png",
                                     FLOOD,
                                     MAP_DOMAIN_NAME[FLOOD],
                                     MAP_SCORE[FLOOD],
                                     MAP_STEP[FLOOD],
                                     rescue_cost,
                                     save_plot=SAVE_RESULT,
                                     consider_cost=False)
  plt.show()
