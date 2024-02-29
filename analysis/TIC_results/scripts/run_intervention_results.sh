# ===== human data models (both: semi-supervised)
# - movers
python analysis/TIC_results/intervention_results.py \
      --domain='movers' --costs='1' --num-runs=100 \
      --dir-name=human_data --is-btil-agent=True \
      --output-name=intv_res_20240213.csv \
      --v-value-file='movers_160_0,30_150_merged_v_values_learned.pickle' \
      --policy1-file='movers_btil_dec_policy_human_woTx_FTTT_160_0,30_a1.npy' \
      --policy2-file='movers_btil_dec_policy_human_woTx_FTTT_160_0,30_a2.npy' \
      --policy3-file='' \
      --tx1-file='movers_btil_dec_tx_human_FTTT_160_0,30_a1.npy' \
      --tx2-file='movers_btil_dec_tx_human_FTTT_160_0,30_a2.npy' \
      --tx3-file='' --bx1-file='' --bx2-file='' --bx3-file=''

# - flood (rescue_2)
python analysis/TIC_results/intervention_results.py \
      --domain='rescue_2' --costs='0' --num-runs=100 \
      --dir-name=human_data --is-btil-agent=True \
      --output-name=intv_res_20240213.csv \
      --v-value-file='rescue_2_160_0,30_30_merged_v_values_learned.pickle' \
      --policy1-file='rescue_2_btil_dec_policy_human_woTx_FTTT_160_0,30_a1.npy' \
      --policy2-file='rescue_2_btil_dec_policy_human_woTx_FTTT_160_0,30_a2.npy' \
      --policy3-file='' \
      --tx1-file='rescue_2_btil_dec_tx_human_FTTT_160_0,30_a1.npy' \
      --tx2-file='rescue_2_btil_dec_tx_human_FTTT_160_0,30_a2.npy' \
      --tx3-file='' --bx1-file='' --bx2-file='' --bx3-file=''

# ===== human data models (human: semi-supervised, robot: unsupervised)
# - movers
# python analysis/TIC_results/intervention_results.py \
#       --domain='movers' --costs='1' --num-runs=100 \
#       --dir-name=human_data --is-btil-agent=True \
#       --output-name=intv_res_20240213_fixedrobot.csv \
#       --v-value-file='movers_160_0,30_150_fixedrobot_merged_v_values_learned.pickle' \
#       --policy1-file='movers_btil_dec_policy_human_woTx_FTTT_160_0,30_a1.npy' \
#       --policy2-file='movers_btil_svi_bx_human_FTTT_160_0,00_a2.npy' \
#       --policy3-file='' \
#       --tx1-file='movers_btil_dec_tx_human_FTTT_160_0,30_a1.npy' \
#       --tx2-file='movers_btil_svi_tx_human_FTTT_160_0,00_a2.npy' \
#       --tx3-file='' --bx1-file='' \
#       --bx2-file='movers_btil_svi_bx_human_FTTT_160_0,00_a2.npy' --bx3-file=''
