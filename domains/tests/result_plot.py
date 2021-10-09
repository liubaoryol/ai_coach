import glob
import os
import random
import pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
  # ##############################################
  # # results
  BETA_PI = False
  TEAM_SYN = True
  INDV_SYN = False
  TEAM_HUM = False
  INDV_HUM = False
  if TEAM_SYN:
    x_true = [0.005454, 0.012632]
    x_sl20_no_Tx = [0.294479, 0.343106]
    x_sl50_no_Tx = [0.260545, 0.280474]
    x_sl100_no_Tx = [0.215499, 0.233522]
    x_sl200_no_Tx = [0.153731, 0.189256]

    x_sl20_w_Tx = [0.332102, 0.278965]
    x_sl50_w_Tx = [0.277737, 0.286407]
    x_sl100_w_Tx = [0.25121, 0.283819]
    x_sl200_w_Tx = [0.176306, 0.232615]

    x_bc20_w_Tx = [0.630276, 0.645003]
    x_bc50_w_Tx = [0.646861, 0.658036]
    x_bc100_w_Tx = [0.646734, 0.646598]
    x_bc200_w_Tx = [0.636374, 0.615177]

    x_semi20_no_Tx = [0.292456, 0.327889]
    x_semi50_no_Tx = [0.26153, 0.275827]
    x_semi100_no_Tx = [0.218926, 0.230714]

    x_semi20_w_Tx = [0.183638, 0.168642]
    x_semi50_w_Tx = [0.167033, 0.159118]
    x_semi100_w_Tx = [0.122282, 0.164532]

    kl_sl20_no_Tx = [0.422531, 0.426059]
    kl_sl50_no_Tx = [0.326864, 0.332515]
    kl_sl100_no_Tx = [0.239354, 0.246812]
    kl_sl200_no_Tx = [0.157211, 0.159222]

    kl_sl20_w_Tx = [0.422531, 0.426059]
    kl_sl50_w_Tx = [0.326864, 0.332515]
    kl_sl100_w_Tx = [0.239354, 0.246812]
    kl_sl200_w_Tx = [0.157211, 0.159222]

    kl_bc20_w_Tx = [0.598566, 0.595354]
    kl_bc50_w_Tx = [0.500704, 0.5036]
    kl_bc100_w_Tx = [0.353092, 0.350601]
    kl_bc200_w_Tx = [0.211326, 0.207852]

    kl_semi20_no_Tx = [0.240487, 0.246558]
    kl_semi50_no_Tx = [0.22093, 0.226812]
    kl_semi100_no_Tx = [0.194589, 0.196721]

    kl_semi20_w_Tx = [0.227926, 0.235615]
    kl_semi50_w_Tx = [0.211874, 0.216018]
    kl_semi100_w_Tx = [0.183715, 0.188997]

    str_title = "Movers and Packers Synthetic"
    # ax1.grid(True)
    # ax2.grid(True)
    sl_x1_Tx = [
        x_sl20_w_Tx[0], x_sl50_w_Tx[0], x_sl100_w_Tx[0], x_sl200_w_Tx[0]
    ]
    sl_x1_noTx = [
        x_sl20_no_Tx[0], x_sl50_no_Tx[0], x_sl100_no_Tx[0], x_sl200_no_Tx[0]
    ]
    bc_x1_Tx = [
        x_bc20_w_Tx[0], x_bc50_w_Tx[0], x_bc100_w_Tx[0], x_bc200_w_Tx[0]
    ]
    semi_x1_Tx = [x_semi20_w_Tx[0], x_semi50_w_Tx[0], x_semi100_w_Tx[0]]
    semi_x1_noTx = [x_semi20_no_Tx[0], x_semi50_no_Tx[0], x_semi100_no_Tx[0]]

    sl_x2_Tx = [
        x_sl20_w_Tx[1], x_sl50_w_Tx[1], x_sl100_w_Tx[1], x_sl200_w_Tx[1]
    ]
    sl_x2_noTx = [
        x_sl20_no_Tx[1], x_sl50_no_Tx[1], x_sl100_no_Tx[1], x_sl200_no_Tx[1]
    ]
    bc_x2_Tx = [
        x_bc20_w_Tx[1], x_bc50_w_Tx[1], x_bc100_w_Tx[1], x_bc200_w_Tx[1]
    ]
    semi_x2_Tx = [x_semi20_w_Tx[1], x_semi50_w_Tx[1], x_semi100_w_Tx[1]]
    semi_x2_noTx = [x_semi20_no_Tx[1], x_semi50_no_Tx[1], x_semi100_no_Tx[1]]

    sl_kl1_noTx = [
        kl_sl20_no_Tx[0], kl_sl50_no_Tx[0], kl_sl100_no_Tx[0], kl_sl200_no_Tx[0]
    ]
    sl_kl2_noTx = [
        kl_sl20_no_Tx[1], kl_sl50_no_Tx[1], kl_sl100_no_Tx[1], kl_sl200_no_Tx[1]
    ]
    semi_kl1_noTx = [
        kl_semi20_no_Tx[0], kl_semi50_no_Tx[0], kl_semi100_no_Tx[0]
    ]
    semi_kl2_noTx = [
        kl_semi20_no_Tx[1], kl_semi50_no_Tx[1], kl_semi100_no_Tx[1]
    ]
    bc_kl1 = [
        kl_bc20_w_Tx[0], kl_bc50_w_Tx[0], kl_bc100_w_Tx[0], kl_bc200_w_Tx[0]
    ]
    bc_kl2 = [
        kl_bc20_w_Tx[1], kl_bc50_w_Tx[1], kl_bc100_w_Tx[1], kl_bc200_w_Tx[1]
    ]
    # sl_xticks = ["20 Samples", "50 Samples", "100 Samples", "200 Samples"]
    sl_ticks = [20, 50, 100, 200]
    # semi_x1_Tx = [x_semi20_w_Tx[0], x_semi50_w_Tx[0], x_semi100_w_Tx[0]]
    # ax1.bar()
    fig1 = plt.figure(figsize=(4.5, 3))
    ax1 = fig1.add_subplot(111)
    ax1.plot(sl_ticks,
             sl_x1_Tx,
             'g.-',
             label="SL w/ Tx",
             clip_on=False,
             fillstyle='none')
    ax1.plot(sl_ticks,
             bc_x1_Tx,
             'b.-',
             label="BC",
             clip_on=False,
             fillstyle='none')
    ax1.plot(sl_ticks,
             sl_x1_noTx,
             'r.-',
             label="SL w/o Tx",
             clip_on=False,
             fillstyle='none')

    ax1n = ax1.twinx()
    ax1n.plot(sl_ticks,
              sl_x2_Tx,
              'g.--',
              label="SL w/ Tx",
              clip_on=False,
              fillstyle='none')
    ax1n.plot(sl_ticks,
              bc_x2_Tx,
              'b.--',
              label="BC",
              clip_on=False,
              fillstyle='none')
    ax1n.plot(sl_ticks,
              sl_x2_noTx,
              'r.--',
              label="SL w/o Tx",
              clip_on=False,
              fillstyle='none')

    FONT_SIZE = 14
    # TITLE_FONT_SIZE = 12
    # LEGENT_FONT_SIZE = 12
    ax1.set_ylabel("Norm. Hamming Dist.", fontsize=FONT_SIZE)
    ax1.set_xlabel("Samples", fontsize=FONT_SIZE)
    ax1.legend(bbox_to_anchor=(1.1, 1), loc="upper left", title="x1")
    ax1n.legend(bbox_to_anchor=(1.1, 0.7), loc="upper left", title="x2")
    ax1.set_title("Latent Inference Performance with Supervised Learning",
                  fontsize=FONT_SIZE)

    fig2 = plt.figure(figsize=(4.5, 3))
    ax2 = fig2.add_subplot(111)
    bar_width = 0.2
    bar_idx = np.arange(3)
    ax2.bar(bar_idx,
            sl_x1_Tx[0:3],
            color='b',
            width=bar_width,
            label="SL w/ Tx")
    bar_idx = bar_idx + bar_width
    ax2.bar(bar_idx,
            semi_x1_Tx[0:3],
            color='g',
            width=bar_width,
            label="Semi w/ Tx")
    bar_idx = bar_idx + bar_width + 0.05
    ax2.bar(bar_idx,
            sl_x1_noTx[0:3],
            color='r',
            width=bar_width,
            label="SL w/o Tx")
    bar_idx = bar_idx + bar_width
    ax2.bar(bar_idx,
            semi_x1_noTx[0:3],
            color='m',
            width=bar_width,
            label="Semi w/o Tx")

    ax2n = ax2.twinx()

    # bar_idx = bar_idx + bar_width + 0.05
    bar_idx = np.arange(3) + 3.5
    ax2n.bar(bar_idx, sl_x2_Tx[0:3], color='b', width=bar_width)
    bar_idx = bar_idx + bar_width
    ax2n.bar(
        bar_idx,
        semi_x2_Tx[0:3],
        color='g',
        width=bar_width,
    )
    bar_idx = bar_idx + bar_width + 0.05
    ax2n.bar(bar_idx, sl_x2_noTx[0:3], color='r', width=bar_width)
    bar_idx = bar_idx + bar_width
    ax2n.bar(bar_idx, semi_x2_noTx[0:3], color='m', width=bar_width)
    ax2.set_xticks([
        1.5 * bar_width + 0.025, 1 + 1.5 * bar_width + 0.025,
        2 + 1.5 * bar_width + 0.025, 3.5 + 1.5 * bar_width + 0.025,
        3.5 + 1 + 1.5 * bar_width + 0.025, 3.5 + 2 + 1.5 * bar_width + 0.025
    ])
    ax2.set_xticklabels(["10%", "25%", "50%", "10%", "25%", "50%"])
    ax2.set_ylabel("Norm. Hamming Dist.", fontsize=FONT_SIZE)
    ax2.set_xlabel("Percentage of Labeled Samples", fontsize=FONT_SIZE)
    ax2.legend(bbox_to_anchor=(1.1, 1), loc="upper left")
    ax2.set_title("Performance with Augmented Unlabeled Samples",
                  fontsize=FONT_SIZE)

    fig3 = plt.figure(figsize=(4.5, 3))
    ax3 = fig3.add_subplot(111)
    bar_width = 0.2
    bar_idx = np.arange(4)
    ax3.bar(bar_idx, bc_kl1, color='black', width=bar_width, label="BC")
    bar_idx = bar_idx + bar_width
    ax3.bar(bar_idx, sl_kl1_noTx, color='b', width=bar_width, label="SL")
    bar_idx = bar_idx + bar_width
    ax3.bar(bar_idx[0:3],
            semi_kl1_noTx[0:3],
            color='g',
            width=bar_width,
            label="Semi")

    x12_space = 4.5
    ax3n = ax3.twinx()
    bar_idx = np.arange(4) + x12_space
    ax3n.bar(bar_idx, bc_kl2, color='black', width=bar_width, label="BC")
    bar_idx = bar_idx + bar_width
    ax3n.bar(bar_idx, sl_kl2_noTx, color='b', width=bar_width, label="SL")
    bar_idx = bar_idx + bar_width
    ax3n.bar(bar_idx[0:3],
             semi_kl2_noTx[0:3],
             color='g',
             width=bar_width,
             label="Semi")
    ax3.set_xticks([
        bar_width, 1 + bar_width, 2 + bar_width, 3 + bar_width,
        x12_space + bar_width, x12_space + 1 + bar_width,
        x12_space + 2 + bar_width, x12_space + 3 + bar_width
    ])
    ax3.set_xticklabels(["20", "50", "100", "200", "20", "50", "100", "200"])
    ax3.set_ylabel("Weighted KL-Divergence.", fontsize=FONT_SIZE)
    ax3.set_xlabel("Number of Labeled Samples", fontsize=FONT_SIZE)
    ax3.legend(bbox_to_anchor=(1.1, 1), loc="upper left")
    ax3.set_title("Policy Learning Performance", fontsize=FONT_SIZE)

    # ax2.set_xticks([20, 50, 100])
    fig1.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()
    plt.show()

  if INDV_SYN:
    x_true = [0.004338, 0]
    x_sl20_no_Tx = [0.527135, 0.45934]
    x_sl50_no_Tx = [0.412589, 0.331087]
    x_sl100_no_Tx = [0.279421, 0.213463]
    x_sl200_no_Tx = [0.174491, 0.112779]

    x_sl20_w_Tx = [0.103468, 0.043543]
    x_sl50_w_Tx = [0.059747, 0.013587]
    x_sl100_w_Tx = [0.0369, 0.009216]
    x_sl200_w_Tx = [0.028663, 0.002439]

    x_bc20_w_Tx = [0.476781, 0.338402]
    x_bc50_w_Tx = [0.436355, 0.331009]
    x_bc100_w_Tx = [0.473739, 0.317923]
    x_bc200_w_Tx = [0.421004, 0.259569]

    x_semi20_no_Tx = [0.430974, 0.380776]
    x_semi50_no_Tx = [0.374241, 0.289578]
    x_semi100_no_Tx = [0.268456, 0.194727]

    x_semi20_w_Tx = [0.108817, 0.056934]
    x_semi50_w_Tx = [0.083477, 0.025131]
    x_semi100_w_Tx = [0.055136, 0.018509]

    kl_sl20_no_Tx = [1.732649, 1.794691]
    kl_sl50_no_Tx = [1.177031, 1.047292]
    kl_sl100_no_Tx = [0.597207, 0.563578]
    kl_sl200_no_Tx = [0.183754, 0.152605]

    kl_sl20_w_Tx = [1.732649, 1.794691]
    kl_sl50_w_Tx = [1.177031, 1.047292]
    kl_sl100_w_Tx = [0.597207, 0.563578]
    kl_sl200_w_Tx = [0.183754, 0.152605]

    kl_bc20_w_Tx = [1.742946, 1.786176]
    kl_bc50_w_Tx = [1.185252, 1.040958]
    kl_bc100_w_Tx = [0.602876, 0.563013]
    kl_bc200_w_Tx = [0.188681, 0.15419]

    kl_semi20_no_Tx = [0.359439, 0.265056]
    kl_semi50_no_Tx = [0.322191, 0.221589]
    kl_semi100_no_Tx = [0.245922, 0.191806]

    kl_semi20_w_Tx = [0.499023, 0.305184]
    kl_semi50_w_Tx = [0.353357, 0.212101]
    kl_semi100_w_Tx = [0.255549, 0.188577]

    # ax1.grid(True)
    # ax2.grid(True)
    sl_x1_Tx = [
        x_sl20_w_Tx[0], x_sl50_w_Tx[0], x_sl100_w_Tx[0], x_sl200_w_Tx[0]
    ]
    sl_x1_noTx = [
        x_sl20_no_Tx[0], x_sl50_no_Tx[0], x_sl100_no_Tx[0], x_sl200_no_Tx[0]
    ]
    bc_x1_Tx = [
        x_bc20_w_Tx[0], x_bc50_w_Tx[0], x_bc100_w_Tx[0], x_bc200_w_Tx[0]
    ]
    semi_x1_Tx = [x_semi20_w_Tx[0], x_semi50_w_Tx[0], x_semi100_w_Tx[0]]
    semi_x1_noTx = [x_semi20_no_Tx[0], x_semi50_no_Tx[0], x_semi100_no_Tx[0]]

    sl_x2_Tx = [
        x_sl20_w_Tx[1], x_sl50_w_Tx[1], x_sl100_w_Tx[1], x_sl200_w_Tx[1]
    ]
    sl_x2_noTx = [
        x_sl20_no_Tx[1], x_sl50_no_Tx[1], x_sl100_no_Tx[1], x_sl200_no_Tx[1]
    ]
    bc_x2_Tx = [
        x_bc20_w_Tx[1], x_bc50_w_Tx[1], x_bc100_w_Tx[1], x_bc200_w_Tx[1]
    ]
    semi_x2_Tx = [x_semi20_w_Tx[1], x_semi50_w_Tx[1], x_semi100_w_Tx[1]]
    semi_x2_noTx = [x_semi20_no_Tx[1], x_semi50_no_Tx[1], x_semi100_no_Tx[1]]

    sl_kl1_noTx = [
        kl_sl20_no_Tx[0], kl_sl50_no_Tx[0], kl_sl100_no_Tx[0], kl_sl200_no_Tx[0]
    ]
    sl_kl2_noTx = [
        kl_sl20_no_Tx[1], kl_sl50_no_Tx[1], kl_sl100_no_Tx[1], kl_sl200_no_Tx[1]
    ]
    semi_kl1_noTx = [
        kl_semi20_no_Tx[0], kl_semi50_no_Tx[0], kl_semi100_no_Tx[0]
    ]
    semi_kl2_noTx = [
        kl_semi20_no_Tx[1], kl_semi50_no_Tx[1], kl_semi100_no_Tx[1]
    ]
    bc_kl1 = [
        kl_bc20_w_Tx[0], kl_bc50_w_Tx[0], kl_bc100_w_Tx[0], kl_bc200_w_Tx[0]
    ]
    bc_kl2 = [
        kl_bc20_w_Tx[1], kl_bc50_w_Tx[1], kl_bc100_w_Tx[1], kl_bc200_w_Tx[1]
    ]
    # sl_xticks = ["20 Samples", "50 Samples", "100 Samples", "200 Samples"]
    sl_ticks = [20, 50, 100, 200]
    # semi_x1_Tx = [x_semi20_w_Tx[0], x_semi50_w_Tx[0], x_semi100_w_Tx[0]]
    # ax1.bar()
    fig1 = plt.figure(figsize=(4.5, 3))
    ax1 = fig1.add_subplot(111)
    ax1.plot(sl_ticks,
             sl_x1_Tx,
             'g.-',
             label="SL w/ Tx",
             clip_on=False,
             fillstyle='none')
    ax1.plot(sl_ticks,
             bc_x1_Tx,
             'b.-',
             label="BC",
             clip_on=False,
             fillstyle='none')
    ax1.plot(sl_ticks,
             sl_x1_noTx,
             'r.-',
             label="SL w/o Tx",
             clip_on=False,
             fillstyle='none')

    ax1n = ax1.twinx()
    ax1n.plot(sl_ticks,
              sl_x2_Tx,
              'g.--',
              label="SL w/ Tx",
              clip_on=False,
              fillstyle='none')
    ax1n.plot(sl_ticks,
              bc_x2_Tx,
              'b.--',
              label="BC",
              clip_on=False,
              fillstyle='none')
    ax1n.plot(sl_ticks,
              sl_x2_noTx,
              'r.--',
              label="SL w/o Tx",
              clip_on=False,
              fillstyle='none')

    FONT_SIZE = 14
    # TITLE_FONT_SIZE = 12
    # LEGENT_FONT_SIZE = 12
    ax1.set_ylabel("Norm. Hamming Dist.", fontsize=FONT_SIZE)
    ax1.set_xlabel("Samples", fontsize=FONT_SIZE)
    ax1.legend(bbox_to_anchor=(1.1, 1), loc="upper left", title="x1")
    ax1n.legend(bbox_to_anchor=(1.1, 0.7), loc="upper left", title="x2")
    ax1.set_title("Latent Inference Performance with Supervised Learning",
                  fontsize=FONT_SIZE)

    fig2 = plt.figure(figsize=(4.5, 3))
    ax2 = fig2.add_subplot(111)
    bar_width = 0.2
    bar_idx = np.arange(3)
    ax2.bar(bar_idx,
            sl_x1_Tx[0:3],
            color='b',
            width=bar_width,
            label="SL w/ Tx")
    bar_idx = bar_idx + bar_width
    ax2.bar(bar_idx,
            semi_x1_Tx[0:3],
            color='g',
            width=bar_width,
            label="Semi w/ Tx")
    bar_idx = bar_idx + bar_width + 0.05
    ax2.bar(bar_idx,
            sl_x1_noTx[0:3],
            color='r',
            width=bar_width,
            label="SL w/o Tx")
    bar_idx = bar_idx + bar_width
    ax2.bar(bar_idx,
            semi_x1_noTx[0:3],
            color='m',
            width=bar_width,
            label="Semi w/o Tx")

    ax2n = ax2.twinx()

    # bar_idx = bar_idx + bar_width + 0.05
    bar_idx = np.arange(3) + 3.5
    ax2n.bar(bar_idx, sl_x2_Tx[0:3], color='b', width=bar_width)
    bar_idx = bar_idx + bar_width
    ax2n.bar(
        bar_idx,
        semi_x2_Tx[0:3],
        color='g',
        width=bar_width,
    )
    bar_idx = bar_idx + bar_width + 0.05
    ax2n.bar(bar_idx, sl_x2_noTx[0:3], color='r', width=bar_width)
    bar_idx = bar_idx + bar_width
    ax2n.bar(bar_idx, semi_x2_noTx[0:3], color='m', width=bar_width)
    ax2.set_xticks([
        1.5 * bar_width + 0.025, 1 + 1.5 * bar_width + 0.025,
        2 + 1.5 * bar_width + 0.025, 3.5 + 1.5 * bar_width + 0.025,
        3.5 + 1 + 1.5 * bar_width + 0.025, 3.5 + 2 + 1.5 * bar_width + 0.025
    ])
    ax2.set_xticklabels(["10%", "25%", "50%", "10%", "25%", "50%"])
    ax2.set_ylabel("Norm. Hamming Dist.", fontsize=FONT_SIZE)
    ax2.set_xlabel("Percentage of Labeled Samples", fontsize=FONT_SIZE)
    ax2.legend(bbox_to_anchor=(1.1, 1), loc="upper left")
    ax2.set_title("Performance with Augmented Unlabeled Samples",
                  fontsize=FONT_SIZE)

    fig3 = plt.figure(figsize=(4.5, 3))
    ax3 = fig3.add_subplot(111)
    bar_width = 0.2
    bar_idx = np.arange(4)
    ax3.bar(bar_idx, bc_kl1, color='black', width=bar_width, label="BC")
    bar_idx = bar_idx + bar_width
    ax3.bar(bar_idx, sl_kl1_noTx, color='b', width=bar_width, label="SL")
    bar_idx = bar_idx + bar_width
    ax3.bar(bar_idx[0:3],
            semi_kl1_noTx[0:3],
            color='g',
            width=bar_width,
            label="Semi")

    x12_space = 4.5
    ax3n = ax3.twinx()
    bar_idx = np.arange(4) + x12_space
    ax3n.bar(bar_idx, bc_kl2, color='black', width=bar_width, label="BC")
    bar_idx = bar_idx + bar_width
    ax3n.bar(bar_idx, sl_kl2_noTx, color='b', width=bar_width, label="SL")
    bar_idx = bar_idx + bar_width
    ax3n.bar(bar_idx[0:3],
             semi_kl2_noTx[0:3],
             color='g',
             width=bar_width,
             label="Semi")
    ax3.set_xticks([
        bar_width, 1 + bar_width, 2 + bar_width, 3 + bar_width,
        x12_space + bar_width, x12_space + 1 + bar_width,
        x12_space + 2 + bar_width, x12_space + 3 + bar_width
    ])
    ax3.set_xticklabels(["20", "50", "100", "200", "20", "50", "100", "200"])
    ax3.set_ylabel("Weighted KL-Divergence.", fontsize=FONT_SIZE)
    ax3.set_xlabel("Number of Labeled Samples", fontsize=FONT_SIZE)
    ax3.legend(bbox_to_anchor=(1.1, 1), loc="upper left")
    ax3.set_title("Policy Learning Performance", fontsize=FONT_SIZE)

    # ax2.set_xticks([20, 50, 100])
    fig1.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()
    plt.show()

  if TEAM_HUM:
    fig = plt.figure(figsize=(7.2, 3))
    str_title = "Movers and Packers Synthetic"

    sl_no_Tx = [0.13301, 0.115342, 0.101212]  # 22, 44, 88
    semi_no_Tx = [0.143632, 0.116562]  # 22, 44

    sl_w_Tx = [0.029247, 0.008197]  # 22, 44
    semi_w_Tx = 0.016588  # 22
    bc_wTx = 0.146157  # 88

  if INDV_HUM:
    fig = plt.figure(figsize=(7.2, 3))
    str_title = "Cleanup"

    sl_no_Tx = [0.431867, 0.345525, 0.282333]  # 28, 55, 110
    semi_no_Tx = [0.413542, 0.332277]  # 28, 55

    sl_w_Tx = [0.189893, 0.092098]  # 22, 44
    semi_w_Tx = 0.179967  # 22
    bc_wTx = 0.486468  # 165

  if BETA_PI:
    fig = plt.figure(figsize=(7.2, 3))
    # str_title = (
    #     "hyperparam: " + str(SEMISUPER_HYPERPARAM) +
    #     ", # labeled: " + str(len(trajectories)) +
    #     ", # unlabeled: " + str(len(unlabeled_traj)))
    str_title = ("KL over beta")
    list_kl1_team = [
        0.211022, 0.209135, 0.198055, 0.164204, 0.157211, 0.157834, 0.166335,
        0.196487
    ]
    list_kl2_team = [
        0.207553, 0.205707, 0.194994, 0.163914, 0.159222, 0.161656, 0.172828,
        0.20674
    ]
    list_kl_idx_team = [1.0001, 1.001, 1.01, 1.1, 1.2, 1.3, 1.5, 2.0]

    list_kl1_indv = [0.188086, 0.185471, 0.183754, 0.225749, 0.340395]
    list_kl2_indv = [0.153631, 0.151331, 0.152605, 0.203895, 0.33548]

    list_kl_idx_indv = [1.0001, 1.001, 1.01, 1.04, 1.1]

    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.grid(True)
    ax2.grid(True)
    ax1.plot(list_kl_idx_team,
             list_kl1_team,
             '.-',
             label="Human",
             clip_on=False,
             fillstyle='none')
    ax1.plot(list_kl_idx_team,
             list_kl2_team,
             '.-',
             label="Robot",
             clip_on=False,
             fillstyle='none')
    ax2.plot(list_kl_idx_indv,
             list_kl1_indv,
             '.-',
             label="Human",
             clip_on=False,
             fillstyle='none')
    ax2.plot(list_kl_idx_indv,
             list_kl2_indv,
             '.-',
             label="Robot",
             clip_on=False,
             fillstyle='none')
    # ax1.axhline(y=full_align_acc1, color='r', linestyle='-', label="SL-Small")
    # ax1.axhline(y=full_align_acc2, color='g', linestyle='-', label="SL-Large")
    FONT_SIZE = 16
    # TITLE_FONT_SIZE = 12
    # LEGENT_FONT_SIZE = 12
    ax1.set_ylabel("KL-Divergence", fontsize=FONT_SIZE)
    ax1.set_xlabel("Beta", fontsize=FONT_SIZE)
    ax1.legend()
    ax1.set_title("Movers and Packers", fontsize=FONT_SIZE)
    ax2.set_ylabel("KL-Divergence", fontsize=FONT_SIZE)
    ax2.set_xlabel("Beta", fontsize=FONT_SIZE)
    ax2.legend()
    ax2.set_title("Cleanup", fontsize=FONT_SIZE)
    plt.show()

  # ax1.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
  # # ax1.set_ylim([70, 100])
  # # ax1.set_xlim([0, 16])
  # ax1.set_title("Full Sequence", fontsize=TITLE_FONT_SIZE)

  #   ax2.plot(part_acc_history,
  #            '.-',
  #            label="SemiSL",
  #            clip_on=False,
  #            fillstyle='none')
  #   if do_sup_infer:
  #     ax2.axhline(y=part_align_acc1, color='r', linestyle='-', label="SL-Small")
  #     ax2.axhline(y=part_align_acc2, color='g', linestyle='-', label="SL-Large")
  #   ax2.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
  #   # ax2.set_ylim([50, 80])
  #   # ax2.set_xlim([0, 16])
  #   ax2.set_title("Partial Sequence (5 Steps)", fontsize=TITLE_FONT_SIZE)
  #   handles, labels = ax2.get_legend_handles_labels()
  #   fig.legend(handles,
  #              labels,
  #              loc='center right',
  #              prop={'size': LEGENT_FONT_SIZE})
  #   fig.text(0.45, 0.04, 'Iteration', ha='center', fontsize=FONT_SIZE)
  #   fig.tight_layout(pad=2.0)
  #   fig.subplots_adjust(right=0.8, bottom=0.2)