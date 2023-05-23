import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="white")


def save_bar_plots_reward(df_input_name, output_name, list_domains,
                          list_domain_names, perfect_scores, perfect_steps,
                          save_plot):
  df = pd.read_csv(df_input_name)

  for idx in range(len(list_domains)):
    df.loc[len(df.index)] = [
        list_domains[idx], "Centralized", 0, 0, 0, perfect_scores[idx],
        perfect_steps[idx]
    ]
    df.loc[len(df.index)] = [
        list_domains[idx], "Centralized", 1, 0, 0, perfect_scores[idx],
        perfect_steps[idx]
    ]

  max_r = 120
  df["real_score"] = max_r + (df["score"] - df["num_feedback"] * df["cost"])
  handle = None
  labels = [
      "Expectation\nValue", "Expectation\nRule", "Confidence\nValue",
      "Confidence\nRule", "No\nintervention", "Centralized\npolicy"
  ]

  fig, axes = plt.subplots(1, 1, figsize=(6, 5))
  for idx in range(len(list_domains)):
    df_domain = df[df["domain"] == list_domains[idx]]

    # ==== reward vs delta =====
    benefit_based = df_domain[((df_domain["strategy"] == "Average")
                               #  | (df_domain["strategy"] == "Argmax")
                               )
                              & (df_domain["cost"] == 0)
                              & (df_domain["interv_thres"] == 5)]
    rule_based = df_domain[(df_domain["strategy"] == "Rule_avg")
                           & (df_domain["cost"] == 0)]
    confidence_value = df_domain[(df_domain["strategy"] == "Argmax_thres")
                                 & (df_domain["cost"] == 0)
                                 & (df_domain["infer_thres"] == 0.3)]
    confidence_rule = df_domain[(df_domain["strategy"] == "Rule_thres")
                                & (df_domain["cost"] == 0)
                                & (df_domain["infer_thres"] == 0.3)]
    baselines = df_domain[((df_domain["strategy"] == "No_intervention")
                           | (df_domain["strategy"] == "Centralized"))
                          & (df_domain["cost"] == 0)]
    methods = pd.concat([
        benefit_based, rule_based, confidence_value, confidence_rule, baselines
    ])

    ax = sns.barplot(ax=axes, data=methods, x='strategy', y='real_score')

    fontsize = 16
    ax.set_ylabel("Task reward", fontsize=fontsize)
    ax.set_xlabel("Strategy", fontsize=fontsize)
    ax.set_xticklabels(labels)
    ax.set_ylim([0, 80])
    list_yticks = ax.get_yticklabels()
    list_yticks_new = [int(ytick.get_text()) - max_r for ytick in list_yticks]
    ax.set_yticklabels(list_yticks_new)
  fig.tight_layout()
  if save_plot:
    fig.savefig(output_name)


def save_bar_plots_objectiveJ(df_input_name, output_name, list_domains,
                              list_domain_names, perfect_scores, perfect_steps,
                              save_plot):
  df = pd.read_csv(df_input_name)

  for idx in range(len(list_domains)):
    df.loc[len(df.index)] = [
        list_domains[idx], "Centralized", 0, 0, 0, perfect_scores[idx],
        perfect_steps[idx]
    ]
    df.loc[len(df.index)] = [
        list_domains[idx], "Centralized", 1, 0, 0, perfect_scores[idx],
        perfect_steps[idx]
    ]

  max_r = 120
  df["real_score"] = max_r + (df["score"] - df["num_feedback"] * df["cost"])
  handle = None
  labels = [
      "Expectation\nValue", "Expectation\nRule", "Confidence\nValue",
      "Confidence\nRule", "No\nintervention", "Centralized\npolicy"
  ]

  fig, axes = plt.subplots(1, 1, figsize=(6, 5))
  for idx in range(len(list_domains)):
    df_domain = df[df["domain"] == list_domains[idx]]

    # ==== reward vs delta =====
    benefit_based = df_domain[((df_domain["strategy"] == "Average")
                               #  | (df_domain["strategy"] == "Argmax")
                               )
                              & (df_domain["cost"] == 1)
                              & (df_domain["interv_thres"] == 5)]
    rule_based = df_domain[(df_domain["strategy"] == "Rule_avg")
                           & (df_domain["cost"] == 1)]
    confidence_value = df_domain[(df_domain["strategy"] == "Argmax_thres")
                                 & (df_domain["cost"] == 1)
                                 & (df_domain["infer_thres"] == 0.3)]
    confidence_rule = df_domain[(df_domain["strategy"] == "Rule_thres")
                                & (df_domain["cost"] == 1)
                                & (df_domain["infer_thres"] == 0.3)]
    baselines = df_domain[((df_domain["strategy"] == "No_intervention")
                           | (df_domain["strategy"] == "Centralized"))
                          & (df_domain["cost"] == 1)]
    methods = pd.concat([
        benefit_based, rule_based, confidence_value, confidence_rule, baselines
    ])

    ax = sns.barplot(ax=axes, data=methods, x='strategy', y='real_score')

    fontsize = 16
    ax.set_ylabel("Objective J", fontsize=fontsize)
    ax.set_xlabel("Strategy", fontsize=fontsize)
    ax.set_xticklabels(labels)
    ax.set_ylim([0, 80])
    list_yticks = ax.get_yticklabels()
    list_yticks_new = [int(ytick.get_text()) - max_r for ytick in list_yticks]
    ax.set_yticklabels(list_yticks_new)
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

  cost = 1

  SAVE_RESULT = True
  NO_SAVE = not SAVE_RESULT
  save_bar_plots_reward(intv_result_name,
                        output_dir + "movers_bar_plots_reward.png",
                        list_domains[:1], list_domain_names[:1],
                        perfect_scores[:1], perfect_steps[:1], SAVE_RESULT)
  save_bar_plots_objectiveJ(intv_result_name,
                            output_dir + "movers_bar_plots_J.png",
                            list_domains[:1], list_domain_names[:1],
                            perfect_scores[:1], perfect_steps[:1], SAVE_RESULT)

  plt.show()
