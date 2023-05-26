import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="white")


def save_bar_plots_score(df_input_name, output_name, domain_name, domain_title,
                         perfect_score, perfect_step, save_plot, use_cost,
                         add_rulebased, add_valuebased, offset, ylim,
                         interv_thres):
  df = pd.read_csv(df_input_name)

  df.loc[len(df.index)] = [
      domain_name, "Centralized", 0, 0, 0, perfect_score, perfect_step
  ]
  df.loc[len(df.index)] = [
      domain_name, "Centralized", 1, 0, 0, perfect_score, perfect_step
  ]

  df_domain = df[df["domain"] == domain_name]

  df_domain["real_score"] = -offset + (
      df_domain["score"] - df_domain["num_feedback"] * df_domain["cost"])
  handle = None

  palette = sns.color_palette()

  colors = [palette[0]]
  labels = ["No\nintervention"]
  if add_rulebased:
    labels.append("Rule-based\nStrategy")
    colors.append(palette[1])
  if add_valuebased:
    labels.append("Value-based\nStrategy")
    colors.append(palette[3])
  labels.append("Centralized\npolicy")
  colors.append(palette[2])

  if use_cost:
    cost = 1
    ylabel = "Objective J"
  else:
    cost = 0
    ylabel = "Task reward"

  figwidth = 6
  if add_rulebased and add_valuebased:
    figwidth = 8

  fig, axes = plt.subplots(1, 1, figsize=(figwidth, 5))

  # ==== reward vs delta =====
  list_methods = []
  baseline_no = df_domain[(df_domain["strategy"] == "No_intervention")
                          & (df_domain["cost"] == cost)]
  list_methods.append(baseline_no)

  if add_rulebased:
    rule_based = df_domain[(df_domain["strategy"] == "Rule_avg")
                           & (df_domain["cost"] == cost)]
    list_methods.append(rule_based)

  if add_valuebased:
    benefit_based = df_domain[(df_domain["strategy"] == "Average")
                              & (df_domain["cost"] == cost)
                              & (df_domain["interv_thres"] == interv_thres)]
    list_methods.append(benefit_based)

  baseline_cen = df_domain[(df_domain["strategy"] == "Centralized")
                           & (df_domain["cost"] == cost)]
  list_methods.append(baseline_cen)

  methods = pd.concat(list_methods)

  ax = sns.barplot(ax=axes,
                   data=methods,
                   x='strategy',
                   y='real_score',
                   palette=colors)

  fontsize = 16
  ax.set_ylabel(ylabel, fontsize=fontsize)
  ax.set_xlabel("Strategy", fontsize=fontsize)
  ax.set_xticklabels(labels)
  ax.set_ylim([0, ylim])
  list_yticks = ax.get_yticklabels()
  list_yticks_new = [int(ytick.get_text()) + offset for ytick in list_yticks]
  ax.set_yticklabels(list_yticks_new)
  ax.set_title(domain_title, fontsize=fontsize + 2)

  fig.tight_layout()
  if save_plot:
    fig.savefig(output_name)


def save_bar_plots_n_feedback(df_input_name, output_name, domain_name,
                              domain_title, perfect_score, perfect_step,
                              save_plot, interv_thres):
  df = pd.read_csv(df_input_name)

  df.loc[len(df.index)] = [
      domain_name, "Centralized", 0, 0, 0, perfect_score, perfect_step
  ]
  df.loc[len(df.index)] = [
      domain_name, "Centralized", 1, 0, 0, perfect_score, perfect_step
  ]

  df_domain = df[df["domain"] == domain_name]

  handle = None

  palette = sns.color_palette()
  colors = [palette[0], palette[3], palette[2]]
  labels = ["No\nintervention"]
  labels.append("Value-based\nStrategy")
  labels.append("Centralized\npolicy")

  cost = 0
  ylabel = "# Intervention"

  fig, axes = plt.subplots(1, 1, figsize=(6, 5))

  # ==== reward vs delta =====
  list_methods = []
  baseline_no = df_domain[(df_domain["strategy"] == "No_intervention")
                          & (df_domain["cost"] == cost)]
  list_methods.append(baseline_no)

  benefit_based = df_domain[(df_domain["strategy"] == "Average")
                            & (df_domain["cost"] == cost)
                            & (df_domain["interv_thres"] == interv_thres)]
  list_methods.append(benefit_based)

  baseline_cen = df_domain[(df_domain["strategy"] == "Centralized")
                           & (df_domain["cost"] == cost)]
  list_methods.append(baseline_cen)

  methods = pd.concat(list_methods)

  ax = sns.barplot(ax=axes,
                   data=methods,
                   x='strategy',
                   y='num_feedback',
                   palette=colors)

  fontsize = 16
  ax.set_ylabel(ylabel, fontsize=fontsize)
  ax.set_xlabel("Strategy", fontsize=fontsize)
  ax.set_xticklabels(labels)
  # list_yticks = ax.get_yticklabels()
  # list_yticks_new = [int(ytick.get_text()) + offset for ytick in list_yticks]
  # ax.set_yticklabels(list_yticks_new)
  ax.set_title(domain_title, fontsize=fontsize + 2)

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

  offsets = [-120, -50, 0, 0]
  ylims = [80, 30, 8, 6]
  interv_thres = [5, 2, 0.1, 0.1]

  cost = 1

  SAVE_RESULT = True
  NO_SAVE = not SAVE_RESULT
  save_bar_plots_score(intv_result_name,
                       output_dir + f"{list_domains[0]}_bar_rule_reward.png",
                       list_domains[0],
                       list_domain_names[0],
                       perfect_scores[0],
                       perfect_steps[0],
                       SAVE_RESULT,
                       use_cost=False,
                       add_rulebased=True,
                       add_valuebased=False,
                       offset=offsets[0],
                       ylim=ylims[0],
                       interv_thres=interv_thres[0])
  save_bar_plots_score(intv_result_name,
                       output_dir + f"{list_domains[0]}_bar_rule_J.png",
                       list_domains[0],
                       list_domain_names[0],
                       perfect_scores[0],
                       perfect_steps[0],
                       SAVE_RESULT,
                       use_cost=True,
                       add_rulebased=True,
                       add_valuebased=False,
                       offset=offsets[0],
                       ylim=ylims[0],
                       interv_thres=interv_thres[0])

  save_bar_plots_score(intv_result_name,
                       output_dir + f"{list_domains[0]}_bar_both_reward.png",
                       list_domains[0],
                       list_domain_names[0],
                       perfect_scores[0],
                       perfect_steps[0],
                       SAVE_RESULT,
                       use_cost=False,
                       add_rulebased=True,
                       add_valuebased=True,
                       offset=offsets[0],
                       ylim=ylims[0],
                       interv_thres=interv_thres[0])
  save_bar_plots_score(intv_result_name,
                       output_dir + f"{list_domains[0]}_bar_both_J.png",
                       list_domains[0],
                       list_domain_names[0],
                       perfect_scores[0],
                       perfect_steps[0],
                       SAVE_RESULT,
                       use_cost=True,
                       add_rulebased=True,
                       add_valuebased=True,
                       offset=offsets[0],
                       ylim=ylims[0],
                       interv_thres=interv_thres[0])

  for idx_dom in [0, 1]:
    save_bar_plots_score(intv_result_name,
                         output_dir +
                         f"{list_domains[idx_dom]}_bar_val_reward.png",
                         list_domains[idx_dom],
                         list_domain_names[idx_dom],
                         perfect_scores[idx_dom],
                         perfect_steps[idx_dom],
                         SAVE_RESULT,
                         use_cost=False,
                         add_rulebased=False,
                         add_valuebased=True,
                         offset=offsets[idx_dom],
                         ylim=ylims[idx_dom],
                         interv_thres=interv_thres[idx_dom])
    save_bar_plots_score(intv_result_name,
                         output_dir + f"{list_domains[idx_dom]}_bar_val_J.png",
                         list_domains[idx_dom],
                         list_domain_names[idx_dom],
                         perfect_scores[idx_dom],
                         perfect_steps[idx_dom],
                         SAVE_RESULT,
                         use_cost=True,
                         add_rulebased=False,
                         add_valuebased=True,
                         offset=offsets[idx_dom],
                         ylim=ylims[idx_dom],
                         interv_thres=interv_thres[idx_dom])

  for idx_dom in [2, 3]:
    save_bar_plots_score(intv_result_name,
                         output_dir +
                         f"{list_domains[idx_dom]}_bar_val_reward.png",
                         list_domains[idx_dom],
                         list_domain_names[idx_dom],
                         perfect_scores[idx_dom],
                         perfect_steps[idx_dom],
                         SAVE_RESULT,
                         use_cost=False,
                         add_rulebased=False,
                         add_valuebased=True,
                         offset=offsets[idx_dom],
                         ylim=ylims[idx_dom],
                         interv_thres=interv_thres[idx_dom])
    save_bar_plots_n_feedback(intv_result_name,
                              output_dir +
                              f"{list_domains[idx_dom]}_bar_val_n_feedback.png",
                              list_domains[idx_dom],
                              list_domain_names[idx_dom],
                              perfect_scores[idx_dom],
                              perfect_steps[idx_dom],
                              SAVE_RESULT,
                              interv_thres=interv_thres[idx_dom])

  plt.show()
