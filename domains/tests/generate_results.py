import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def get_df_data(df, str_prefix, measure_name):
  list_queried = []

  list_numbers = [20, 50, 100, 200]
  for num in list_numbers:
    str_index = str_prefix + " " + str(num)
    if str_index not in df.index:
      break

    list_queried.append(df.loc[str_index][measure_name])

  return list_queried


def plot_bar(fig,
             subplot_idx,
             dict_list,
             domain,
             sl_algo,
             semi_algo,
             met_a1,
             met_a2,
             x_label,
             y_label,
             title,
             font_size=14):
  ax1 = fig.add_subplot(subplot_idx)
  bar_width = 0.3
  bar_idx = np.arange(4)
  ax1.bar(bar_idx,
          dict_list[domain + sl_algo + met_a1],
          color='b',
          label='BTIL-sup(Alice)',
          width=bar_width)
  bar_idx = bar_idx + bar_width
  ax1.bar(bar_idx[:-1],
          dict_list[domain + semi_algo + met_a1],
          color='g',
          label='BTIL-semi(Alice)',
          width=bar_width)

  ax1.axvline(x=len(bar_idx) - 0.5, color='k', linestyle='--', lw=1)
  bar_idx = np.arange(4) + len(bar_idx)
  ax1.bar(bar_idx,
          dict_list[domain + sl_algo + met_a2],
          color='r',
          label='BTIL-sup(Rob)',
          width=bar_width)
  bar_idx = bar_idx + bar_width
  ax1.bar(bar_idx[:-1],
          dict_list[domain + semi_algo + met_a2],
          color='m',
          label='BTIL-semi(Rob)',
          width=bar_width)

  xticks = [
      0.5 * bar_width, 1 + 0.5 * bar_width, 2 + 0.5 * bar_width,
      3 + 0.5 * bar_width, 4 + 0.5 * bar_width, 5 + 0.5 * bar_width,
      6 + 0.5 * bar_width, 7 + 0.5 * bar_width
  ]
  xtick_labels = ["10%", "25%", "50%", "100%", "10%", "25%", "50%", "100%"]
  ax1.set_title(title, fontsize=font_size)
  ax1.set_xticks(xticks)
  ax1.set_xticklabels(xtick_labels)

  ax1.set_ylabel(y_label, fontsize=font_size)
  ax1.set_xlabel(x_label, fontsize=font_size)

  return ax1


if __name__ == "__main__":
  team_data_csv = os.path.join(os.path.dirname(__file__), "team_data.csv")
  team_df = pd.read_csv(team_data_csv, index_col=0)

  indv_data_csv = os.path.join(os.path.dirname(__file__), "indv_data.csv")
  indv_df = pd.read_csv(indv_data_csv, index_col=0)

  dict_list = {}
  list_metric = ['x1 mean', 'x2 mean', 'weighted JS1', 'weighted JS2']
  list_algo = ['SL with Tx', 'Semi with Tx', 'SL w/o Tx', 'Semi w/o Tx']
  list_domain = ['team', 'indv']
  for domain in list_domain:
    df = team_df if domain == 'team' else indv_df

    for algo in list_algo:
      for metric in list_metric:
        dict_list[domain + algo + metric] = get_df_data(df, algo, metric)

  font_size = 14

  fig1 = plt.figure(figsize=(8, 6))
  ax1 = plot_bar(fig1, 221, dict_list, list_domain[0], list_algo[0],
                 list_algo[1], list_metric[0], list_metric[1], "",
                 "Norm. Hamming Dist.", "With Tx", font_size)
  ax2 = plot_bar(fig1, 223, dict_list, list_domain[0], list_algo[0],
                 list_algo[1], list_metric[2], list_metric[3], "", "JS Div.",
                 "", font_size)
  ax3 = plot_bar(fig1, 222, dict_list, list_domain[0], list_algo[2],
                 list_algo[3], list_metric[0], list_metric[1], "", "", "W/o Tx",
                 font_size)
  ax4 = plot_bar(fig1, 224, dict_list, list_domain[0], list_algo[2],
                 list_algo[3], list_metric[2], list_metric[3], "", "", "",
                 font_size)
  #   fig1.supxlabel("Percentage of Labeled Samples")
  handles, labels = ax1.get_legend_handles_labels()
  ax1.set_ylim([0, 0.4])
  ax3.set_ylim([0, 0.4])
  ax2.set_ylim([0, 0.1])
  ax4.set_ylim([0, 0.1])
  fig1.text(0.5,
            0.07,
            'Percentage of Labeled Samples',
            ha='center',
            fontsize=font_size)
  fig1.legend(handles,
              labels,
              loc='lower center',
              ncol=4,
              bbox_to_anchor=(0.5, 0.0))
  fig1.tight_layout()
  fig1.subplots_adjust(bottom=0.15)

  fig2 = plt.figure(figsize=(8, 6))
  ax5 = plot_bar(fig2, 221, dict_list, list_domain[1], list_algo[0],
                 list_algo[1], list_metric[0], list_metric[1], "",
                 "Norm. Hamming Dist.", "With Tx", font_size)
  ax6 = plot_bar(fig2, 223, dict_list, list_domain[1], list_algo[0],
                 list_algo[1], list_metric[2], list_metric[3], "", "JS Div.",
                 "", font_size)
  ax7 = plot_bar(fig2, 222, dict_list, list_domain[1], list_algo[2],
                 list_algo[3], list_metric[0], list_metric[1], "", ".",
                 "W/o Tx", font_size)
  ax8 = plot_bar(fig2, 224, dict_list, list_domain[1], list_algo[2],
                 list_algo[3], list_metric[2], list_metric[3], "", ".", "",
                 font_size)
  handles, labels = ax5.get_legend_handles_labels()
  ax5.set_ylim([0, 0.6])
  ax7.set_ylim([0, 0.6])
  ax6.set_ylim([0, 0.2])
  ax8.set_ylim([0, 0.2])
  fig2.text(0.5,
            0.07,
            'Percentage of Labeled Samples',
            ha='center',
            fontsize=font_size)
  fig2.legend(handles,
              labels,
              loc='lower center',
              ncol=4,
              bbox_to_anchor=(0.5, 0.0))

  fig2.tight_layout()
  fig2.subplots_adjust(bottom=0.15)

  fig1.savefig("box results.png")
  fig2.savefig("bag results.png")
  plt.show()
