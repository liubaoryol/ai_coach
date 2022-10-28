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
  h, l = ax.get_legend_handles_labels()
  ax.legend(h, ["150(100%)", "500(30%)", "500(100%)"],
            title="# Data (Supervision)")

  randomguess = [0.25, 0.25, 0.20, 0.33]
  for i in range(4):
    ax.hlines(y=randomguess[i],
              xmin=i - 0.5,
              xmax=i + 0.5,
              color='black',
              ls='--')

  fontsize = 13
  ax.set_xlabel(None)
  ax.set_ylabel("Accuracy", fontsize=fontsize)
  ax.set_ylim([0, 1])
  ax.tick_params(axis='x', which='major', labelsize=fontsize)

  fig = ax.get_figure()
  fig.tight_layout()
  if save_plot:
    fig.savefig(output_name)


def save_score_vs_delta_plots(df_input_name, output_name, list_domains,
                              list_domain_names, save_plot):
  df = pd.read_csv(df_input_name)

  num_domain = len(list_domains)

  fig, axes = plt.subplots(1, num_domain, figsize=(5 * num_domain, 5))
  for idx in range(len(list_domains)):
    df_domain = df[df["domain"] == list_domains[idx]]
    score_vs_delta = df_domain[((df_domain["strategy"] == "Average")
                                | (df_domain["strategy"] == "Argmax"))
                               & (df_domain["interv_thres"] != 0)]
    all_interv_arg = df_domain[(df_domain["strategy"] == "Argmax")
                               & (df_domain["interv_thres"] == 0)]
    no_interv = df_domain[(df_domain["strategy"] == "No_intervention")]
    all_interv_arg_score = all_interv_arg["score"].mean()
    no_interv_score = no_interv["score"].mean()

    ax = sns.lineplot(ax=axes[idx],
                      data=score_vs_delta,
                      x='interv_thres',
                      y='score',
                      hue="strategy")
    ax.axhline(all_interv_arg_score, label="Everytime", color='g')
    ax.axhline(no_interv_score, label="None", color='black')

    h, l = ax.get_legend_handles_labels()
    ax.set_ylabel("Score")
    ax.set_xlabel(r"Benefit threshold ($\delta$)" +
                  f"\n{list_domain_names[idx]}")
    ax.legend(h, l + ["Everytime", "None"], title="Strategy")
  fig.tight_layout()
  if save_plot:
    fig.savefig(output_name)


def save_score_vs_delta_plots2(df_input_name, output_name, list_domains,
                               list_domain_names, perfect_scores, perfect_steps,
                               save_plot):
  df = pd.read_csv(df_input_name)

  num_domain = len(list_domains)

  fig, axes = plt.subplots(1, num_domain, figsize=(5 * num_domain, 4))
  for idx in range(len(list_domains)):
    df_domain = df[df["domain"] == list_domains[idx]]
    score_vs_delta = df_domain[((df_domain["strategy"] == "Average")
                                | (df_domain["strategy"] == "Argmax"))]

    # all_interv_arg = df_domain[(df_domain["strategy"] == "Argmax")
    #                            & (df_domain["interv_thres"] == 0)]
    no_interv = df_domain[(df_domain["strategy"] == "No_intervention")]
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
                             & (df_domain["infer_thres"] == 0)]
      rule_based_score = rule_based["score"].mean()
      ax.axhline(rule_based_score, label="Rule-based", color='c')
      labels.append("Rule-based")

    perfect_scr = perfect_scores[idx]
    ax.axhline(perfect_scr, label="Perfect", color='green', ls='-.')
    # ax.axhline(all_interv_arg_score, label="Everytime", color='g')
    ax.axhline(no_interv_score, label="None", color='black', ls='--')
    labels.append("Centralized policy")
    labels.append("No intervention")

    h, l = ax.get_legend_handles_labels()
    fontsize = 15
    ax.set_ylabel("Score", fontsize=fontsize)
    ax.set_xlabel(r"Benefit threshold ($\delta$)" +
                  f"\n{list_domain_names[idx]}",
                  fontsize=fontsize)
    ax.legend(h, labels, title="Strategy", loc="upper right")
    # ax.tick_params(axis='x', which='major', labelsize=15)
  fig.tight_layout()
  if save_plot:
    fig.savefig(output_name)


def save_score_vs_theta_plots(df_input_name, output_name, list_domains,
                              list_domain_names, save_plot):
  df = pd.read_csv(df_input_name)

  num_domain = len(list_domains)

  fig, axes = plt.subplots(1, num_domain, figsize=(5 * num_domain, 5))
  axes = axes if num_domain > 1 else [axes]
  for idx in range(len(list_domains)):
    df_domain = df[df["domain"] == list_domains[idx]]
    score_vs_theta = df_domain[(df_domain["strategy"] == "Argmax_thres")
                               & (df_domain["infer_thres"] != 0)]
    all_interv_arg = df_domain[(df_domain["strategy"] == "Argmax_thres")
                               & (df_domain["infer_thres"] == 0)]
    no_interv = df_domain[(df_domain["strategy"] == "No_intervention")]
    all_interv_arg_score = all_interv_arg["score"].mean()
    no_interv_score = no_interv["score"].mean()

    ax = sns.lineplot(ax=axes[idx],
                      data=score_vs_theta,
                      x='infer_thres',
                      y='score',
                      hue="strategy")
    ax.axhline(all_interv_arg_score, label="Everytime", color='g')
    ax.axhline(no_interv_score, label="None", color='black')

    h, l = ax.get_legend_handles_labels()
    ax.set_ylabel("Score")
    ax.set_xlabel(r"Inference threshold ($\theta$)"
                  f"\n{list_domain_names[idx]}")
    ax.legend(h, ["Argmax", "Everytime", "None"], title="Strategy")
  fig.tight_layout()
  if save_plot:
    fig.savefig(output_name)


def save_score_vs_theta_plots2(df_input_name, output_name, list_domains,
                               list_domain_names, perfect_scores, perfect_steps,
                               save_plot):
  df = pd.read_csv(df_input_name)

  num_domain = len(list_domains)

  fig, axes = plt.subplots(1, num_domain, figsize=(5 * num_domain, 3))
  axes = axes if num_domain > 1 else [axes]
  for idx in range(len(list_domains)):
    df_domain = df[df["domain"] == list_domains[idx]]
    score_vs_theta = df_domain[(df_domain["strategy"] == "Argmax_thres")]
    no_interv = df_domain[(df_domain["strategy"] == "No_intervention")]
    no_interv_score = no_interv["score"].mean()

    ax = sns.lineplot(ax=axes[idx],
                      data=score_vs_theta,
                      x='infer_thres',
                      y='score',
                      hue="strategy")

    perfect_scr = perfect_scores[idx]
    ax.axhline(perfect_scr, label="Perfect", color='y')
    ax.axhline(no_interv_score, label="None", color='black')

    h, l = ax.get_legend_handles_labels()
    ax.set_ylabel("Score")
    ax.set_xlabel(r"Inference threshold ($\theta$)"
                  f"\n{list_domain_names[idx]}")
    ax.legend(h, ["Argmax", "Perfect", "None"], title="Strategy")
  fig.tight_layout()
  if save_plot:
    fig.savefig(output_name)


def save_num_feedback_vs_delta_plots(df_input_name, output_name, list_domains,
                                     list_domain_names, save_plot):
  df = pd.read_csv(df_input_name)

  num_domain = len(list_domains)

  fig, axes = plt.subplots(1, num_domain, figsize=(5 * num_domain, 3))
  axes = axes if num_domain > 1 else [axes]
  for idx in range(len(list_domains)):
    df_domain = df[df["domain"] == list_domains[idx]]
    score_vs_delta = df_domain[((df_domain["strategy"] == "Average")
                                | (df_domain["strategy"] == "Argmax"))
                               & (df_domain["interv_thres"] != 0)]

    ax = sns.lineplot(ax=axes[idx],
                      data=score_vs_delta,
                      x='interv_thres',
                      y='num_feedback',
                      hue="strategy")

    h, l = ax.get_legend_handles_labels()
    ax.set_ylabel("# intervention")
    ax.set_xlabel(r"Benefit threshold ($\delta$)" +
                  f"\n{list_domain_names[idx]}")
    ax.legend(h, l, title="Strategy")

  fig.tight_layout()
  if save_plot:
    fig.savefig(output_name)


def save_score_vs_intervention_plots(df_input_name, output_name, list_domains,
                                     list_domain_names, save_plot):
  df = pd.read_csv(df_input_name)

  num_domain = len(list_domains)

  fig, axes = plt.subplots(1, num_domain, figsize=(5 * num_domain, 3))
  axes = axes if num_domain > 1 else [axes]
  for idx in range(len(list_domains)):
    df_domain = df[df["domain"] == list_domains[idx]]
    df_domain = df_domain.replace(to_replace="Argmax_thres",
                                  value='Argmax',
                                  regex=False)
    df_domain = df_domain.replace(to_replace="Rule_thres",
                                  value='Rule (Threshold)',
                                  regex=False)
    df_domain = df_domain.replace(to_replace="Rule_avg",
                                  value='Rule (Avg.)',
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

    h, l = ax.get_legend_handles_labels()
    ax.set_xlabel("# intervention")
    ax.set_ylabel("Score")
    ax.legend(h, l, title="Strategy")

  fig.tight_layout()
  if save_plot:
    fig.savefig(output_name)


if __name__ == "__main__":
  data_dir = os.path.join(os.path.dirname(__file__), "data/")
  output_dir = os.path.join(os.path.dirname(__file__), "output/")

  eval_result_name = data_dir + "eval_result3.csv"
  intv_result_name = data_dir + "intervention_result4.csv"

  list_domains = ["movers", "cleanup_v3", "rescue_2", "rescue_3"]
  list_domain_names = ["Movers", "Cleanup", "Flood", "Blackout"]

  perfect_scores = [-43, -21, 7, 5]
  perfect_steps = [43, 21, 19, 9]

  SAVE_RESULT = True
  NO_SAVE = not SAVE_RESULT

  # save_evaluation_plot(eval_result_name, output_dir + "inference_eval.png",
  #                      list_domain_names, SAVE_RESULT)
  save_score_vs_delta_plots2(intv_result_name,
                             output_dir + "score_vs_delta.png", list_domains,
                             list_domain_names, perfect_scores, perfect_steps,
                             SAVE_RESULT)
  # save_score_vs_theta_plots2(intv_result_name,
  #                            output_dir + "score_vs_theta.png",
  #                            list_domains[:1], list_domain_names,
  #                            perfect_scores, perfect_steps, NO_SAVE)
  # save_num_feedback_vs_delta_plots(intv_result_name,
  #                                  output_dir + "num_feedback_vs_delta.png",
  #                                  list_domains[:1], list_domain_names, NO_SAVE)
  # save_score_vs_intervention_plots(intv_result_name,
  #                                  output_dir + "num_feedback_vs_score.png",
  #                                  list_domains[:1], list_domain_names, NO_SAVE)

  plt.show()
