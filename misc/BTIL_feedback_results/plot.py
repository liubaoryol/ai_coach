import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="white")


def save_evaluation_plot(df_input_name, output_name, save_plot):
  df = pd.read_csv(df_input_name)
  ax = sns.barplot(x='domain', y='value', hue='train_setup', data=df)

  ax.set_xticklabels(["Movers", "Cleanup", "Rescue", "Rescue2"])
  h, l = ax.get_legend_handles_labels()
  ax.legend(h, ["150(100%)", "500(30%)", "500(100%)"],
            title="# Data (Supervision)")
  ax.set_xlabel(None)
  ax.set_ylabel("Accuracy")
  ax.set_ylim([0, 1])
  ax.tick_params(axis='x', which='major', labelsize=12)

  fig = ax.get_figure()
  fig.tight_layout()
  if save_plot:
    fig.savefig(output_name)


def save_score_vs_delta_plots(df_input_name, output_name, save_plot):
  df = pd.read_csv(df_input_name)

  list_domains = ["movers", "cleanup_v2", "rescue_2"]
  list_domain_names = ["Movers", "Cleanup", "Rescue"]
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


def save_score_vs_theta_plots(df_input_name, output_name, save_plot):
  df = pd.read_csv(df_input_name)

  list_domains = ["movers"]
  list_domain_names = ["Movers"]
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


def save_num_feedback_vs_delta_plots(df_input_name, output_name, save_plot):
  df = pd.read_csv(df_input_name)

  list_domains = ["movers"]
  list_domain_names = ["Movers"]
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


def save_score_vs_intervention_plots(df_input_name, output_name, save_plot):
  df = pd.read_csv(df_input_name)

  # list_domains = ["movers", "cleanup_v2", "rescue_2"]
  list_domains = ["movers"]
  list_domain_names = ["Movers"]
  num_domain = len(list_domains)

  fig, axes = plt.subplots(1, num_domain, figsize=(5 * num_domain, 5))
  axes = axes if num_domain > 1 else [axes]
  for idx in range(len(list_domains)):
    df_domain = df[df["domain"] == list_domains[idx]]
    df_domain = df_domain.replace(to_replace="Argmax_thres",
                                  value='Argmax',
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
    ax.legend(h, ["Argmax", "Average", "No intervention"], title="Strategy")

  fig.tight_layout()
  if save_plot:
    fig.savefig(output_name)


if __name__ == "__main__":
  data_dir = os.path.join(os.path.dirname(__file__), "data/")
  output_dir = os.path.join(os.path.dirname(__file__), "output/")

  eval_result_name = data_dir + "eval_result2.csv"
  intv_result_name = data_dir + "intervention_result2.csv"

  SAVE_RESULT = True
  NO_SAVE = not SAVE_RESULT

  # save_evaluation_plot(eval_result_name, output_dir + "inference_eval.png",
  #                      NO_SAVE)
  save_score_vs_delta_plots(intv_result_name, output_dir + "score_vs_delta.png",
                            NO_SAVE)
  save_score_vs_theta_plots(intv_result_name, output_dir + "score_vs_theta.png",
                            NO_SAVE)
  save_num_feedback_vs_delta_plots(intv_result_name,
                                   output_dir + "num_feedback_vs_delta.png",
                                   NO_SAVE)
  save_score_vs_intervention_plots(intv_result_name,
                                   output_dir + "num_feedback_vs_score.png",
                                   NO_SAVE)

  plt.show()
