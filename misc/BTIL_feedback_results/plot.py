import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="white")


def save_evaluation_plot(df_input_name, output_name, list_domain_names,
                         save_plot):
  df = pd.read_csv(df_input_name)
  fig, axes = plt.subplots(1, 1, figsize=(7, 5))
  ax = sns.barplot(ax=axes, x='domain', y='value', hue='train_setup', data=df)

  ax.set_xticklabels(list_domain_names)
  h, _ = ax.get_legend_handles_labels()
  ax.legend(h, ["150(100%)", "500(30%)", "500(100%)"],
            title="# Data (Supervision)")

  randomguess = [0.25, 0.20, 0.25, 0.33]
  for i in range(4):
    ax.hlines(y=randomguess[i],
              xmin=i - 0.5,
              xmax=i + 0.5,
              color='black',
              ls='--')

  fontsize = 14
  ax.set_xlabel(None)
  ax.set_ylabel("Accuracy", fontsize=fontsize)
  ax.set_ylim([0, 1])
  ax.tick_params(axis='x', which='major', labelsize=fontsize)

  fig = ax.get_figure()
  fig.tight_layout()
  if save_plot:
    fig.savefig(output_name)


def save_box_plots(df_input_name, output_name, list_domains, list_domain_names,
                   perfect_scores, perfect_steps, save_plot):
  df = pd.read_csv(df_input_name)

  df["real_score"] = df["score"] - df["num_feedback"] * df["cost"]

  # num_domain = len(list_domains)

  handle = None
  labels = [
      "Expectation-Value", "Deterministic-Value", "Expectation-Rule",
      "Centralized policy", "No intervention"
  ]
  fig, axes = plt.subplots(2, 2, figsize=(10, 8))
  for idx in range(len(list_domains)):
    df_domain = df[df["domain"] == list_domains[idx]]
    # ==== reward vs delta =====
    reward_vs_delta = df_domain[((df_domain["strategy"] == "Average")
                                 | (df_domain["strategy"] == "Argmax"))
                                & (df_domain["cost"] == 0)]
    assert len(reward_vs_delta) == 1800

    # all_interv_arg = df_domain[(df_domain["strategy"] == "Argmax")
    #                            & (df_domain["interv_thres"] == 0)]
    no_interv = df_domain[(df_domain["strategy"] == "No_intervention")
                          & (df_domain["cost"] == 0)]
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
                             & (df_domain["cost"] == 0)]
      assert len(rule_based) == 100
      rule_based_score = rule_based["score"].mean()
      ax.axhline(rule_based_score,
                 label="Rule-Average",
                 color='c',
                 lw=line_width)

    perfect_scr = perfect_scores[idx]
    ax.axhline(perfect_scr,
               label="Perfect",
               color='green',
               ls='-.',
               lw=line_width)
    ax.axhline(no_interv_score,
               label="None",
               color='black',
               ls='--',
               lw=line_width)

    h, _ = ax.get_legend_handles_labels()
    fontsize = 16
    ax.set_ylabel("Task reward", fontsize=fontsize)
    ax.set_xlabel(r"Benefit threshold ($\delta$)")
    ax.set_xlabel(r"Benefit threshold ($\delta$)", fontsize=fontsize)
    ax.set_title(list_domain_names[idx], fontsize=fontsize + 2)
    if list_domains[idx] == "movers":
      handle = h
    if handle is None:
      handle = h
    ax.legend([], [], frameon=False)

    # ==== score vs delta =====
    score_vs_delta = df_domain[((df_domain["strategy"] == "Average")
                                | (df_domain["strategy"] == "Argmax"))
                               & (df_domain["cost"] == 1)]
    assert len(score_vs_delta) == 1800

    # no_interv = df_domain[(df_domain["strategy"] == "No_intervention")
    #                       & (df_domain["cost"] == 1)]
    # assert len(no_interv) == 100
    # no_interv_score = no_interv["real_score"].mean()

    ax = sns.lineplot(ax=axes[1][idx],
                      data=score_vs_delta,
                      x='interv_thres',
                      y='real_score',
                      hue="strategy",
                      lw=line_width)
    if list_domains[idx] == "movers":
      rule_based = df_domain[(df_domain["strategy"] == "Rule_avg")
                             & (df_domain["cost"] == 1)]
      assert len(rule_based) == 100
      rule_based_score = rule_based["real_score"].mean()
      ax.axhline(rule_based_score,
                 label="Rule-Average",
                 color='c',
                 lw=line_width)

    perfect_scr = perfect_scores[idx] - 1 * perfect_steps[idx]
    ax.axhline(perfect_scr,
               label="Perfect",
               color='green',
               ls='-.',
               lw=line_width)
    ax.axhline(no_interv_score,
               label="None",
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


def save_rescue_plots(df_input_name, output_name, list_domains,
                      list_domain_names, perfect_scores, perfect_steps,
                      save_plot):
  df = pd.read_csv(df_input_name)

  df["real_score"] = df["score"] - df["num_feedback"] * df["cost"]

  handle = None
  labels = [
      "Expectation-Value", "Deterministic-Value", "Centralized policy",
      "No intervention"
  ]
  fig, axes = plt.subplots(2, 2, figsize=(10, 8))
  for idx in range(len(list_domains)):
    df_domain = df[df["domain"] == list_domains[idx]]
    # ==== reward vs delta =====
    reward_vs_delta = df_domain[((df_domain["strategy"] == "Average")
                                 | (df_domain["strategy"] == "Argmax"))
                                & (df_domain["cost"] == 0)]
    assert len(reward_vs_delta) == 1800

    # all_interv_arg = df_domain[(df_domain["strategy"] == "Argmax")
    #                            & (df_domain["interv_thres"] == 0)]
    no_interv = df_domain[(df_domain["strategy"] == "No_intervention")
                          & (df_domain["cost"] == 0)]
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
               label="Perfect",
               color='green',
               ls='-.',
               lw=line_width)
    ax.axhline(no_interv_score,
               label="None",
               color='black',
               ls='--',
               lw=line_width)

    h, _ = ax.get_legend_handles_labels()
    fontsize = 16
    ax.set_ylabel("Task reward", fontsize=fontsize)
    ax.set_xlabel(r"Benefit threshold ($\delta$)")
    ax.set_xlabel(r"Benefit threshold ($\delta$)", fontsize=fontsize)
    ax.set_title(list_domain_names[idx], fontsize=fontsize + 2)
    if handle is None:
      handle = h
    ax.legend([], [], frameon=False)

    # ==== num intervention vs delta =====
    num_feedback_vs_delta = df_domain[((df_domain["strategy"] == "Average")
                                       | (df_domain["strategy"] == "Argmax"))
                                      & (df_domain["cost"] == 1)]
    assert len(num_feedback_vs_delta) == 1800

    ax = sns.lineplot(ax=axes[1][idx],
                      data=num_feedback_vs_delta,
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


def save_reward_vs_delta_plots(df_input_name,
                               output_name,
                               list_domains,
                               list_domain_names,
                               perfect_scores,
                               perfect_steps,
                               cost,
                               save_plot,
                               show_legend=True):
  df = pd.read_csv(df_input_name)

  num_domain = len(list_domains)

  if show_legend:
    hie = 5
  else:
    hie = 4

  handle = None
  label = None
  fig, axes = plt.subplots(1, num_domain, figsize=(5 * num_domain, hie))
  for idx in range(len(list_domains)):
    df_domain = df[df["domain"] == list_domains[idx]]
    score_vs_delta = df_domain[((df_domain["strategy"] == "Average")
                                | (df_domain["strategy"] == "Argmax"))
                               & (df_domain["cost"] == cost)]
    assert len(score_vs_delta) == 1800

    # all_interv_arg = df_domain[(df_domain["strategy"] == "Argmax")
    #                            & (df_domain["interv_thres"] == 0)]
    no_interv = df_domain[(df_domain["strategy"] == "No_intervention")
                          & (df_domain["cost"] == cost)]
    assert len(no_interv) == 100
    # all_interv_arg_score = all_interv_arg["score"].mean()
    no_interv_score = no_interv["score"].mean()

    labels = ["Value-Average", "Value-Threshold"]
    ax = sns.lineplot(ax=axes[idx],
                      data=score_vs_delta,
                      x='interv_thres',
                      y='score',
                      hue="strategy")
    if list_domains[idx] == "movers":
      rule_based = df_domain[(df_domain["strategy"] == "Rule_thres")
                             & (df_domain["infer_thres"] == 0) &
                             (df_domain["cost"] == cost)]
      assert len(rule_based) == 100
      rule_based_score = rule_based["score"].mean()
      ax.axhline(rule_based_score, label="Rule-based", color='c')
      labels.append("Rule-based")

    perfect_scr = perfect_scores[idx]
    ax.axhline(perfect_scr, label="Perfect", color='green', ls='-.')
    # ax.axhline(all_interv_arg_score, label="Everytime", color='g')
    ax.axhline(no_interv_score, label="None", color='black', ls='--')
    labels.append("Centralized policy")
    labels.append("No intervention")

    h, _ = ax.get_legend_handles_labels()
    fontsize = 18
    ax.set_ylabel("Task Reward", fontsize=fontsize)
    ax.set_xlabel(r"Benefit threshold ($\delta$)", fontsize=fontsize)
    ax.set_title(f"\n{list_domain_names[idx]}", fontsize=fontsize)
    # ax.set_xlabel(r"Benefit threshold ($\delta$)" +
    #               f"\n{list_domain_names[idx]}",
    #               fontsize=fontsize)
    # ax.legend(h, labels, title="Strategy", loc="upper right")
    # ax.tick_params(axis='x', which='major', labelsize=15)
    if list_domains[idx] == "movers":
      handle = h
      label = labels
    if handle is None:
      handle = h
      label = labels
    ax.legend([], [], frameon=False)
  if show_legend:
    fig.legend(handle,
               label,
               loc='lower center',
               ncol=3,
               bbox_to_anchor=(0.5, 0.0),
               prop={'size': 15})
  fig.tight_layout()
  if show_legend:
    fig.subplots_adjust(bottom=0.3)
  if save_plot:
    fig.savefig(output_name)


def save_score_vs_delta_plots(df_input_name, output_name, list_domains,
                              list_domain_names, perfect_scores, perfect_steps,
                              cost, save_plot):
  df = pd.read_csv(df_input_name)

  df["real_score"] = df["score"] - df["num_feedback"] * df["cost"]

  num_domain = len(list_domains)

  handle = None
  label = None
  fig, axes = plt.subplots(1, num_domain, figsize=(5 * num_domain, 4.5))
  for idx in range(len(list_domains)):
    df_domain = df[df["domain"] == list_domains[idx]]
    score_vs_delta = df_domain[((df_domain["strategy"] == "Average")
                                | (df_domain["strategy"] == "Argmax"))
                               & (df_domain["cost"] == cost)]
    assert len(score_vs_delta) == 1800

    # all_interv_arg = df_domain[(df_domain["strategy"] == "Argmax")
    #                            & (df_domain["interv_thres"] == 0)]
    no_interv = df_domain[(df_domain["strategy"] == "No_intervention")
                          & (df_domain["cost"] == cost)]
    assert len(no_interv) == 100
    # all_interv_arg_score = all_interv_arg["score"].mean()
    no_interv_score = no_interv["real_score"].mean()

    labels = ["Value-Average", "Value-Threshold"]
    ax = sns.lineplot(ax=axes[idx],
                      data=score_vs_delta,
                      x='interv_thres',
                      y='real_score',
                      hue="strategy")
    if list_domains[idx] == "movers":
      rule_based = df_domain[(df_domain["strategy"] == "Rule_avg")
                             #  & (df_domain["infer_thres"] == 0)
                             & (df_domain["cost"] == cost)]
      assert len(rule_based) == 100
      rule_based_score = rule_based["real_score"].mean()
      ax.axhline(rule_based_score, label="Rule-Average", color='c')
      labels.append("Rule-Average")

    perfect_scr = perfect_scores[idx] - cost * perfect_steps[idx]
    ax.axhline(perfect_scr, label="Perfect", color='green', ls='-.')
    # ax.axhline(all_interv_arg_score, label="Everytime", color='g')
    ax.axhline(no_interv_score, label="None", color='black', ls='--')
    labels.append("Centralized policy")
    labels.append("No intervention")

    h, _ = ax.get_legend_handles_labels()
    fontsize = 18
    ax.set_ylabel("Objective(J)", fontsize=fontsize)
    ax.set_xlabel(r"Benefit threshold ($\delta$)")
    ax.set_xlabel(r"Benefit threshold ($\delta$)", fontsize=fontsize)
    ax.set_title(f"\n{list_domain_names[idx]}", fontsize=fontsize)
    if list_domains[idx] == "movers":
      handle = h
      label = labels
    if handle is None:
      handle = h
      label = labels
    ax.legend([], [], frameon=False)
  fig.legend(handle,
             label,
             loc='lower center',
             ncol=3,
             bbox_to_anchor=(0.5, 0.0),
             prop={'size': 15})
  fig.tight_layout()
  fig.subplots_adjust(bottom=0.3)
  if save_plot:
    fig.savefig(output_name)


def save_reward_vs_theta_plots(df_input_name, output_name, list_domains,
                               list_domain_names, perfect_scores, perfect_steps,
                               save_plot):
  df = pd.read_csv(df_input_name)

  num_domain = len(list_domains)

  fig, axes = plt.subplots(1, num_domain, figsize=(5 * num_domain, 4))
  axes = axes if num_domain > 1 else [axes]
  for idx in range(len(list_domains)):
    df_domain = df[df["domain"] == list_domains[idx]]
    score_vs_theta = df_domain[((df_domain["strategy"] == "Argmax_thres")
                                | (df_domain["strategy"] == "Rule_thres"))
                               & (df_domain["cost"] == 0)]
    assert len(score_vs_theta) == 1200
    no_interv = df_domain[(df_domain["strategy"] == "No_intervention")
                          & (df_domain["cost"] == 0)]
    assert len(no_interv) == 100
    no_interv_score = no_interv["score"].mean()

    ax = sns.lineplot(ax=axes[idx],
                      data=score_vs_theta,
                      x='infer_thres',
                      y='score',
                      hue="strategy")

    perfect_scr = perfect_scores[idx]
    ax.axhline(perfect_scr, label="Perfect", color='y', ls='-.')
    ax.axhline(no_interv_score, label="None", color='black', ls='--')

    h, _ = ax.get_legend_handles_labels()
    ax.set_ylabel("Reward")
    ax.set_xlabel(r"Inference threshold ($\theta$)")
    ax.set_title(f"\n{list_domain_names[idx]}")
    ax.legend(h, [
        "Value-Threshold", "Rule-Threshold", "Centralized policy",
        "No intervention"
    ])
  fig.tight_layout()
  if save_plot:
    fig.savefig(output_name)


def save_score_vs_theta_plots(df_input_name, output_name, list_domains,
                              list_domain_names, perfect_scores, perfect_steps,
                              cost, dict_interv_thres, save_plot):
  df = pd.read_csv(df_input_name)
  df["real_score"] = df["score"] - df["num_feedback"] * df["cost"]

  num_domain = len(list_domains)

  fig, axes = plt.subplots(1, num_domain, figsize=(5 * num_domain, 4))
  axes = axes if num_domain > 1 else [axes]
  for idx in range(len(list_domains)):
    df_domain = df[df["domain"] == list_domains[idx]]
    score_vs_theta = df_domain[((df_domain["strategy"] == "Argmax_thres")
                                | (df_domain["strategy"] == "Rule_thres"))
                               & (df_domain["cost"] == cost)]
    assert len(score_vs_theta) == 1200

    no_interv = df_domain[(df_domain["strategy"] == "No_intervention")
                          & (df_domain["cost"] == cost)]
    assert len(no_interv) == 100
    no_interv_score = no_interv["real_score"].mean()

    avg_rule = df_domain[(df_domain["strategy"] == "Rule_avg")
                         & (df_domain["cost"] == cost)]
    assert len(avg_rule) == 100
    avg_rule_score = avg_rule["real_score"].mean()

    avg_value = df_domain[(df_domain["strategy"] == "Average")
                          & (df_domain["interv_thres"] == 5)
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
               label="Value-Average",
               color='m',
               ls=':',
               lw=line_width)
    ax.axhline(avg_rule_score,
               label="Rule-Average",
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
               label="None",
               color='black',
               ls='--',
               lw=line_width)

    thr = dict_interv_thres[list_domains[idx]][3]
    h, _ = ax.get_legend_handles_labels()
    fontsize = 13
    ax.set_ylabel("Objective(J)", fontsize=fontsize)
    ax.set_xlabel(r"Inference threshold ($\theta$)", fontsize=fontsize)
    ax.set_title(list_domain_names[idx], fontsize=fontsize + 2)
    ax.legend(h, [
        r"Confidence-Value($\delta=$" + f"{thr})", "Confidence-Rule",
        r"Expectation-Value($\delta=$" + f"{thr})", "Expectation-Rule",
        "Centralized policy", "No intervention"
    ],
              prop={'size': 8})
  fig.tight_layout()
  if save_plot:
    fig.savefig(output_name)


def save_num_feedback_vs_delta_plots(df_input_name, output_name, list_domains,
                                     list_domain_names, cost, save_plot):
  df = pd.read_csv(df_input_name)

  num_domain = len(list_domains)

  fig, axes = plt.subplots(1, num_domain, figsize=(5 * num_domain, 3))
  axes = axes if num_domain > 1 else [axes]
  for idx in range(len(list_domains)):
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


def save_reward_vs_intervention_plots(df_input_name, output_name, list_domains,
                                      list_domain_names, cost, save_plot):
  df = pd.read_csv(df_input_name)
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
                                  value=r'Value-Threshold($\delta=5$)',
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


def save_score_vs_intervention_plots(df_input_name, output_name, list_domains,
                                     list_domain_names, perfect_scores,
                                     perfect_steps, cost, dict_interv_thres,
                                     save_plot):
  df = pd.read_csv(df_input_name)
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
    thr = dict_interv_thres[list_domains[idx]][3]
    df_domain = df_domain.replace(to_replace="Argmax",
                                  value='Deterministic-Value',
                                  regex=False)
    df_domain = df_domain.replace(to_replace="Argmax_thres",
                                  value=r'Confidence-Value($\delta=$' +
                                  f"{thr})",
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
  data_dir = os.path.join(os.path.dirname(__file__), "data/")
  output_dir = os.path.join(os.path.dirname(__file__), "output/")

  eval_result_name = data_dir + "eval_result3.csv"
  intv_result_name = data_dir + "intervention_result8.csv"

  list_domains = ["movers", "cleanup_v3", "rescue_2", "rescue_3"]
  list_domain_names = ["Movers", "Cleanup", "Flood", "Blackout"]

  dict_interv_thres = {
      list_domains[0]: [0, 1, 3, 5, 10, 15, 20, 30, 50],
      list_domains[1]: [0, 0.3, 0.5, 1, 2, 5, 10, 15, 20],
      list_domains[2]: [0, 0.1, 0.3, 0.5, 1, 1.5, 2, 3, 5],
      list_domains[3]: [0, 0.1, 0.2, 0.3, 0.5, 1, 1.5, 2, 3],
  }

  perfect_scores = [-43, -21, 7, 5]
  perfect_steps = [43, 21, 19, 9]

  budget_scores = [-86, -42, 3, 3]
  budget_steps = [86, 42, 30, 15]
  cost = 1

  SAVE_RESULT = True
  NO_SAVE = not SAVE_RESULT

  # save_evaluation_plot(eval_result_name, output_dir + "inference_eval.png",
  #                      list_domain_names, SAVE_RESULT)

  save_box_plots(intv_result_name, output_dir + "delta_box.png",
                 list_domains[:2], list_domain_names[:2], perfect_scores[:2],
                 perfect_steps[:2], SAVE_RESULT)
  save_rescue_plots(intv_result_name, output_dir + "delta_rescue.png",
                    list_domains[2:], list_domain_names[2:], perfect_scores[2:],
                    perfect_steps[2:], SAVE_RESULT)

  # save_reward_vs_delta_plots(intv_result_name,
  #                            output_dir + "reward_vs_delta_box.png",
  #                            list_domains[:2], list_domain_names[:2],
  #                            perfect_scores[:2], perfect_steps[:2], 0,
  #                            SAVE_RESULT, False)
  # save_score_vs_delta_plots(intv_result_name,
  #                           output_dir + "newscore_vs_delta_box.png",
  #                           list_domains[:2], list_domain_names[:2],
  #                           perfect_scores[:2], perfect_steps[:2], cost,
  #                           SAVE_RESULT)
  # save_reward_vs_delta_plots(intv_result_name,
  #                            output_dir + "reward_vs_delta_rescue.png",
  #                            list_domains[2:], list_domain_names[2:],
  #                            perfect_scores[2:], perfect_steps[2:], 0,
  #                            SAVE_RESULT, True)
  # save_num_feedback_vs_delta_plots(
  #     intv_result_name, output_dir + "num_feedback_vs_delta_rescue.png",
  #     list_domains[2:], list_domain_names[2:], cost, SAVE_RESULT)
  # save_reward_vs_theta_plots(intv_result_name,
  #                            output_dir + "reward_vs_theta.png",
  #                            list_domains[:1], list_domain_names[:1],
  #                            perfect_scores[:1], perfect_steps[:1],
  #                            SAVE_RESULT)
  save_score_vs_theta_plots(intv_result_name, output_dir + "score_vs_theta.png",
                            list_domains[:1], list_domain_names[:1],
                            perfect_scores[:1], perfect_steps[:1], cost,
                            dict_interv_thres, SAVE_RESULT)
  # save_num_feedback_vs_delta_plots(intv_result_name,
  #                                  output_dir + "num_feedback_vs_delta.png",
  #                                  list_domains, list_domain_names, cost,
  #                                  SAVE_RESULT)

  # save_reward_vs_intervention_plots(intv_result_name,
  #                                   output_dir + "num_feedback_vs_reward.png",
  #                                   list_domains[:1], list_domain_names[:1],
  #                                   0, SAVE_RESULT)
  save_score_vs_intervention_plots(intv_result_name,
                                   output_dir + "num_feedback_vs_score.png",
                                   list_domains[:1], list_domain_names[:1],
                                   perfect_scores[:1], perfect_steps[:1], 1,
                                   dict_interv_thres, SAVE_RESULT)

  plt.show()
