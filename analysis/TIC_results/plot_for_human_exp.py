import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="white")


def save_box_plots(df,
                   output_name,
                   list_domains,
                   list_domain_names,
                   perfect_scores,
                   perfect_steps,
                   save_plot,
                   cost=1):

  df["real_score"] = df["score"] - df["num_feedback"] * df["cost"]

  # num_domain = len(list_domains)

  handle = None
  fig, axes = plt.subplots(2, 2, figsize=(10, 8))
  for idx in range(len(list_domains)):
    df_domain = df[df["domain"] == list_domains[idx]]
    # ==== reward vs delta =====
    reward_vs_delta = df_domain[(
        (df_domain["strategy"] == "Average")
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
    ax = sns.lineplot(ax=axes[0][idx],
                      data=reward_vs_delta,
                      x='interv_thres',
                      y='score',
                      hue="strategy",
                      lw=line_width)
    if list_domains[idx] == "movers":
      rule_based = df_domain[(df_domain["strategy"] == "Rule_avg")
                             & (df_domain["cost"] == cost)]
      assert len(rule_based) == 100
      rule_based_score = rule_based["score"].mean()
      ax.axhline(rule_based_score,
                 label="Expectation-Rule",
                 color='c',
                 lw=line_width)

    perfect_scr = perfect_scores[idx]
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
    ax.set_title(list_domain_names[idx], fontsize=fontsize + 2)
    handle = h
    ax.legend([], [], frameon=False)

    # ==== score vs delta =====
    # score_vs_delta = df_domain[((df_domain["strategy"] == "Average")
    #                             | (df_domain["strategy"] == "Argmax")
    #                             | (df_domain["strategy"] == "Argmax_robot_fix"))
    #                            & (df_domain["cost"] == cost)]
    # assert len(score_vs_delta) == 2700

    # no_interv = df_domain[(df_domain["strategy"] == "No_intervention")
    #                       & (df_domain["cost"] == 1)]
    # assert len(no_interv) == 100
    # no_interv_score = no_interv["real_score"].mean()

    ax = sns.lineplot(ax=axes[1][idx],
                      data=reward_vs_delta,
                      x='interv_thres',
                      y='real_score',
                      hue="strategy",
                      lw=line_width)
    if list_domains[idx] == "movers":
      rule_based = df_domain[(df_domain["strategy"] == "Rule_avg")
                             & (df_domain["cost"] == cost)]
      assert len(rule_based) == 100
      rule_based_score = rule_based["real_score"].mean()
      ax.axhline(rule_based_score,
                 label="Expectation-Rule",
                 color='c',
                 lw=line_width)

    perfect_scr = perfect_scores[idx] - 1 * perfect_steps[idx]
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
    ax.set_title(list_domain_names[idx], fontsize=fontsize + 2)
    ax.legend([], [], frameon=False)
  fig.legend(handle,
             labels,
             loc='lower center',
             ncol=5,
             bbox_to_anchor=(0.5, 0.0),
             columnspacing=0.8,
             prop={'size': 12})
  fig.tight_layout()
  fig.subplots_adjust(bottom=0.14)
  if save_plot:
    fig.savefig(output_name)


def save_rescue_plots(df,
                      output_name,
                      list_domains,
                      list_domain_names,
                      perfect_scores,
                      perfect_steps,
                      save_plot,
                      cost=1):

  df["real_score"] = df["score"] - df["num_feedback"] * df["cost"]

  handle = None
  fig, axes = plt.subplots(2, 2, figsize=(10, 8))
  for idx in range(len(list_domains)):
    df_domain = df[df["domain"] == list_domains[idx]]
    # ==== reward vs delta =====
    reward_vs_delta = df_domain[(
        (df_domain["strategy"] == "Average")
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
    ax = sns.lineplot(ax=axes[0][idx],
                      data=reward_vs_delta,
                      x='interv_thres',
                      y='score',
                      hue="strategy",
                      lw=line_width)

    perfect_scr = perfect_scores[idx]
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
    ax.set_title(list_domain_names[idx], fontsize=fontsize + 2)
    if handle is None:
      handle = h
    ax.legend([], [], frameon=False)

    # ==== num intervention vs delta =====
    # num_feedback_vs_delta = df_domain[((df_domain["strategy"] == "Average")
    #                                    | (df_domain["strategy"] == "Argmax"))
    #                                   & (df_domain["cost"] == cost)]
    # assert len(num_feedback_vs_delta) == 1800

    ax = sns.lineplot(ax=axes[1][idx],
                      data=reward_vs_delta,
                      x='interv_thres',
                      y='num_feedback',
                      hue="strategy",
                      lw=line_width)

    ax.set_ylabel("# intervention", fontsize=fontsize)
    ax.set_title(list_domain_names[idx], fontsize=fontsize + 2)
    ax.set_xlabel(r"Benefit threshold ($\delta$)", fontsize=fontsize)
    ax.legend([], [], frameon=False)

  fig.legend(handle,
             labels,
             loc='lower center',
             ncol=4,
             columnspacing=1.3,
             bbox_to_anchor=(0.5, 0.0),
             prop={'size': 15})
  fig.tight_layout()
  fig.subplots_adjust(bottom=0.14)
  if save_plot:
    fig.savefig(output_name)


def save_reward_vs_theta_plots(df, output_name, list_domains, list_domain_names,
                               perfect_scores, perfect_steps, save_plot,
                               list_interv_thres, list_cost):

  num_domain = len(list_domains)

  fig, axes = plt.subplots(1, num_domain, figsize=(5 * num_domain, 4))
  axes = axes if num_domain > 1 else [axes]
  for idx in range(len(list_domains)):
    cost = list_cost[idx]
    interv_thres = list_interv_thres[idx]
    df_domain = df[df["domain"] == list_domains[idx]]
    score_vs_theta = df_domain[(
        ((df_domain["strategy"] == "Argmax_thres") &
         (df_domain["interv_thres"] == interv_thres))
        | ((df_domain["strategy"] == "Argmax_thres_robot_fix") &
           (df_domain["interv_thres"] == interv_thres))
        | (df_domain["strategy"] == "Rule_thres"))
                               & (df_domain["cost"] == cost)]
    assert len(score_vs_theta) == 1800
    no_interv = df_domain[(df_domain["strategy"] == "No_intervention")
                          & (df_domain["cost"] == cost)]
    assert len(no_interv) == 100
    no_interv_score = no_interv["score"].mean()

    ax = sns.lineplot(ax=axes[idx],
                      data=score_vs_theta,
                      x='infer_thres',
                      y='score',
                      hue="strategy")

    perfect_scr = perfect_scores[idx]
    ax.axhline(perfect_scr, label="Centralized policy", color='y', ls='-.')
    ax.axhline(no_interv_score, label="No intervention", color='black', ls='--')

    h, lables = ax.get_legend_handles_labels()
    lables[lables.index("Argmax_thres")] = f"Value-Threshold {interv_thres}"
    lables[lables.index("Rule_thres")] = "Rule-Threshold"
    lables[lables.index("Argmax_robot_fix")] = (
        f"Value-Thres-FixRobot {interv_thres}")

    ax.set_ylabel("Reward")
    ax.set_xlabel(r"Inference threshold ($\theta$)")
    ax.set_title(f"\n{list_domain_names[idx]}")
    ax.legend(h, lables)
  fig.tight_layout()
  if save_plot:
    fig.savefig(output_name)


def save_score_vs_theta_plots(df, output_name, list_domains, list_domain_names,
                              perfect_scores, perfect_steps, list_cost,
                              list_interv_thres, save_plot):
  df["real_score"] = df["score"] - df["num_feedback"] * df["cost"]

  num_domain = len(list_domains)

  fig, axes = plt.subplots(1, num_domain, figsize=(5 * num_domain, 4))
  axes = axes if num_domain > 1 else [axes]
  for idx in range(len(list_domains)):
    cost = list_cost[idx]
    interv_thres = list_interv_thres[idx]
    df_domain = df[df["domain"] == list_domains[idx]]
    score_vs_theta = df_domain[(
        ((df_domain["strategy"] == "Argmax_thres") &
         (df_domain["interv_thres"] == interv_thres))
        | ((df_domain["strategy"] == "Argmax_thres_robot_fix") &
           (df_domain["interv_thres"] == interv_thres))
        | (df_domain["strategy"] == "Rule_thres"))
                               & (df_domain["cost"] == cost)]
    assert len(score_vs_theta) == 1800

    no_interv = df_domain[(df_domain["strategy"] == "No_intervention")
                          & (df_domain["cost"] == cost)]
    assert len(no_interv) == 100
    no_interv_score = no_interv["real_score"].mean()

    avg_rule = df_domain[(df_domain["strategy"] == "Rule_avg")
                         & (df_domain["cost"] == cost)]
    assert len(avg_rule) == 100
    avg_rule_score = avg_rule["real_score"].mean()

    avg_value = df_domain[(df_domain["strategy"] == "Average")
                          & (df_domain["interv_thres"] == interv_thres)
                          & (df_domain["cost"] == cost)]
    assert len(avg_value) == 100
    avg_value_score = avg_value["real_score"].mean()

    ax = sns.lineplot(ax=axes[idx],
                      data=score_vs_theta,
                      x='infer_thres',
                      y='real_score',
                      hue="strategy")
    line_width = 2
    ax.axhline(avg_value_score,
               label=r"Expectation-Value($\delta=$" + f"{interv_thres})",
               color='m',
               ls=':',
               lw=line_width)
    ax.axhline(avg_rule_score,
               label="Expectation-Rule",
               color='c',
               ls='-',
               lw=line_width)
    perfect_scr = perfect_scores[idx] - cost * perfect_steps[idx]
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

    h, lables = ax.get_legend_handles_labels()
    lables[lables.index("Argmax_thres")] = (r"Confidence-Value($\delta=$" +
                                            f"{interv_thres})")
    lables[lables.index("Rule_thres")] = "Confidence-Rule"
    lables[lables.index("Argmax_thres_robot_fix")] = (
        f"Conf-Value-FixRobot {interv_thres}")

    fontsize = 13
    ax.set_ylabel("Objective(J)", fontsize=fontsize)
    ax.set_xlabel(r"Inference threshold ($\theta$)", fontsize=fontsize)
    ax.set_title(list_domain_names[idx], fontsize=fontsize + 2)
    ax.legend(h, lables, prop={'size': 8})
  fig.tight_layout()
  if save_plot:
    fig.savefig(output_name)


def save_num_feedback_vs_delta_plots(df, output_name, list_domains,
                                     list_domain_names, list_cost, save_plot):

  num_domain = len(list_domains)

  fig, axes = plt.subplots(1, num_domain, figsize=(5 * num_domain, 3))
  axes = axes if num_domain > 1 else [axes]
  for idx in range(len(list_domains)):
    cost = list_cost[idx]
    df_domain = df[df["domain"] == list_domains[idx]]
    score_vs_delta = df_domain[((df_domain["strategy"] == "Average")
                                | (df_domain["strategy"] == "Argmax"))
                               & (df_domain["cost"] == cost)]
    assert len(score_vs_delta) == 1800

    ax = sns.lineplot(ax=axes[idx],
                      data=score_vs_delta,
                      x='interv_thres',
                      y='num_feedback',
                      hue="strategy")

    h, _ = ax.get_legend_handles_labels()
    ax.set_ylabel("# intervention")
    ax.set_title(list_domain_names[idx])
    ax.set_xlabel(r"Benefit threshold ($\delta$)")
    ax.legend(h, ["Value-Average", "Value-Threshold"])

  fig.tight_layout()
  if save_plot:
    fig.savefig(output_name)


def save_reward_vs_intervention_plots(df, output_name, list_domains,
                                      list_domain_names, cost, save_plot):
  # df["real_score"] = df["score"] - df["num_feedback"] * df["cost"]

  print(len(df))
  df = df[(df.strategy != "Rule_thres_budget")
          & (df.strategy != "Rule_avg_budget") &
          (df.strategy != "No_intervention_budget") &
          (df.strategy != "Average_budget") & (df.strategy != "Argmax_budget") &
          (df.strategy != "Argmax_thres_budget") & (df.cost == cost)]
  print(len(df))

  num_domain = len(list_domains)

  fig, axes = plt.subplots(1, num_domain, figsize=(5 * num_domain, 4))
  axes = axes if num_domain > 1 else [axes]
  for idx in range(len(list_domains)):
    df_domain = df[df["domain"] == list_domains[idx]]
    print(len(df_domain))
    df_domain = df_domain.replace(to_replace="Argmax",
                                  value='Value-Deterministic',
                                  regex=False)
    df_domain = df_domain.replace(to_replace="Argmax_thres",
                                  value='Value-Threshold',
                                  regex=False)
    df_domain = df_domain.replace(to_replace="Average",
                                  value='Value-Average',
                                  regex=False)
    df_domain = df_domain.replace(to_replace="Rule_thres",
                                  value='Rule-Threshold',
                                  regex=False)
    df_domain = df_domain.replace(to_replace="Rule_avg",
                                  value='Rule-Average',
                                  regex=False)
    df_domain = df_domain.replace(to_replace="No_intervention",
                                  value='No intervention',
                                  regex=False)

    df_mean = df_domain.groupby(['strategy', 'interv_thres',
                                 'infer_thres'])[["score",
                                                  "num_feedback"]].mean()

    ax = sns.scatterplot(ax=axes[idx],
                         data=df_mean,
                         x='num_feedback',
                         y='score',
                         hue='strategy',
                         style='strategy')

    fontsize = 13
    h, lable = ax.get_legend_handles_labels()
    ax.set_xlabel("# intervention", fontsize=fontsize)
    ax.set_ylabel("Task reward", fontsize=fontsize)
    ax.set_title(list_domain_names[idx], fontsize=fontsize + 2)
    ax.legend(h, lable)

  fig.tight_layout()
  if save_plot:
    fig.savefig(output_name)


def save_score_vs_intervention_plots(df, output_name, list_domains,
                                     list_domain_names, perfect_scores,
                                     perfect_steps, cost, save_plot):
  df["real_score"] = df["score"] - df["num_feedback"] * df["cost"]

  print(len(df))
  df = df[(df.strategy != "Rule_thres_budget")
          & (df.strategy != "Rule_avg_budget") &
          (df.strategy != "No_intervention_budget") &
          (df.strategy != "Average_budget") & (df.strategy != "Argmax_budget") &
          (df.strategy != "Argmax_thres_budget") & (df.cost == cost)]
  print(len(df))

  num_domain = len(list_domains)

  fig, axes = plt.subplots(1, num_domain, figsize=(5 * num_domain, 4))
  axes = axes if num_domain > 1 else [axes]
  for idx in range(len(list_domains)):
    df_domain = df[df["domain"] == list_domains[idx]]
    print(len(df_domain))
    df_domain = df_domain.replace(to_replace="Argmax",
                                  value='Deterministic-Value',
                                  regex=False)
    df_domain = df_domain.replace(to_replace="Argmax_thres",
                                  value='Confidence-Value',
                                  regex=False)
    df_domain = df_domain.replace(to_replace="Average",
                                  value='Expectation-Value',
                                  regex=False)
    df_domain = df_domain.replace(to_replace="Rule_thres",
                                  value='Confidence-Rule',
                                  regex=False)
    df_domain = df_domain.replace(to_replace="Rule_avg",
                                  value='Expectation-Rule',
                                  regex=False)
    df_domain = df_domain.replace(to_replace="No_intervention",
                                  value='No intervention',
                                  regex=False)

    df_mean = df_domain.groupby(['strategy', 'interv_thres',
                                 'infer_thres'])[["real_score",
                                                  "num_feedback"]].mean()

    ax = sns.scatterplot(ax=axes[idx],
                         data=df_mean,
                         x='num_feedback',
                         y='real_score',
                         hue='strategy',
                         style='strategy',
                         s=70)

    perfect_scr = perfect_scores[idx] - cost * perfect_steps[idx]
    ax.plot(perfect_steps[idx], perfect_scr, 'c^', label="Centralized policy")

    fontsize = 13
    h, label = ax.get_legend_handles_labels()
    ax.set_xlabel("# intervention", fontsize=fontsize)
    ax.set_ylabel("Objective(J)", fontsize=fontsize)
    ax.set_title(list_domain_names[idx], fontsize=fontsize + 2)
    ax.legend(h, label, prop={'size': 8})

  fig.tight_layout()
  if save_plot:
    fig.savefig(output_name)


if __name__ == "__main__":
  data_dir = os.path.join(os.path.dirname(__file__), "human_data/")
  output_dir = os.path.join(os.path.dirname(__file__), "human_output/")

  MOVERS = "movers"
  CLEANUP = "cleanup_v3"
  FLOOD = "rescue_2"
  BLACKOUT = "rescue_3"

  list_domains = [MOVERS, FLOOD]

  intv_result_name = "intervention_results_20240213"
  list_intv_files = [
      intv_result_name + f"-{dname}.csv" for dname in list_domains
  ]

  prefix = "unsup_robot_"

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

  cost = 1

  SAVE_RESULT = True
  NO_SAVE = not SAVE_RESULT

  df_intv_res = pd.concat(map(pd.read_csv, list_intv_files), ignore_index=True)

  list_domains = [MOVERS]
  save_box_plots(df_intv_res,
                 output_dir + prefix + "delta_box.png",
                 list_domains,
                 [MAP_DOMAIN_NAME[dname] for dname in list_domains],
                 [MAP_SCORE[dname] for dname in list_domains],
                 [MAP_STEP[dname] for dname in list_domains],
                 SAVE_RESULT,
                 cost=cost)

  list_domains = [FLOOD]
  save_rescue_plots(df_intv_res,
                    output_dir + prefix + "delta_rescue.png",
                    list_domains,
                    [MAP_DOMAIN_NAME[dname] for dname in list_domains],
                    [MAP_SCORE[dname] for dname in list_domains],
                    [MAP_STEP[dname] for dname in list_domains],
                    SAVE_RESULT,
                    cost=cost)

  list_domains = [MOVERS]
  save_score_vs_theta_plots(df_intv_res,
                            output_dir + prefix + "score_vs_theta.png",
                            list_domains,
                            [MAP_DOMAIN_NAME[dname] for dname in list_domains],
                            [MAP_SCORE[dname] for dname in list_domains],
                            [MAP_STEP[dname] for dname in list_domains],
                            list_cost=[cost],
                            list_interv_thres=[0],
                            save_plot=SAVE_RESULT)

  plt.show()
