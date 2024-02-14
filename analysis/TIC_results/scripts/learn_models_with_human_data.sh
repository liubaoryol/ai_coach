# ===== semi-supervised learning with human data
# - movers
python analysis/TIC_results/two_agent_domain_save_models.py \
      --domain=movers --synthetic=False --num-training-data=160 \
      --supervsion=0.3 --gen-trainset=False --batch-size=-1 \
      --beta-pi=0.01 --beta-tx=0.01 --tx-dependency=FTTT

# - flood (rescue_2)
python analysis/TIC_results/two_agent_domain_save_models.py \
      --domain=rescue_2 --synthetic=False --num-training-data=160 \
      --supervsion=0.3 --gen-trainset=False --batch-size=-1 \
      --beta-pi=0.01 --beta-tx=0.01 --tx-dependency=FTTT

# ===== unsupervised learning with robot data
# - movers
python analysis/TIC_results/two_agent_domain_save_models.py \
      --domain=movers --synthetic=False --num-training-data=160 \
      --supervsion=0.0 --gen-trainset=False --batch-size=80 \
      --beta-pi=0.01 --beta-tx=0.01 --tx-dependency=FTTT

# - flood (rescue_2)
python analysis/TIC_results/two_agent_domain_save_models.py \
      --domain=rescue_2 --synthetic=False --num-training-data=160 \
      --supervsion=0.0 --gen-trainset=False --batch-size=80 \
      --beta-pi=0.01 --beta-tx=0.01 --tx-dependency=FTTT

# ===== generate v-value (human: semi-supervised, robot: semi-supervised)
# - movers
python analysis/TIC_results/save_merged_v_values.py \
      --domain=movers --iteration=150 --output-suffix='' \
      --save-dir=human_data \
      --policy1-file='movers_btil_dec_policy_human_woTx_FTTT_160_0,30_a1.npy' \
      --policy2-file='movers_btil_dec_policy_human_woTx_FTTT_160_0,30_a2.npy' \
      --tx1-file='movers_btil_dec_tx_human_FTTT_160_0,30_a1.npy' \
      --tx2-file='movers_btil_dec_tx_human_FTTT_160_0,30_a2.npy'

# - flood (rescue_2)
python analysis/TIC_results/save_merged_v_values.py \
      --domain=rescue_2 --iteration=30 --output-suffix='' \
      --save-dir=human_data \
      --policy1-file='rescue_2_btil_dec_policy_human_woTx_FTTT_160_0,30_a1.npy' \
      --policy2-file='rescue_2_btil_dec_policy_human_woTx_FTTT_160_0,30_a2.npy' \
      --tx1-file='rescue_2_btil_dec_tx_human_FTTT_160_0,30_a1.npy' \
      --tx2-file='rescue_2_btil_dec_tx_human_FTTT_160_0,30_a2.npy'

# ===== generate v-value (human: semi-supervised, robot: unsupervised)
# - movers
python analysis/TIC_results/save_merged_v_values.py \
      --domain=movers --iteration=150 --output-suffix='fixedrobot' \
      --save-dir=human_data \
      --policy1-file='movers_btil_dec_policy_human_woTx_FTTT_160_0,30_a1.npy' \
      --policy2-file='movers_btil_svi_policy_human_woTx_FTTT_160_0,00_a2.npy' \
      --tx1-file='movers_btil_dec_tx_human_FTTT_160_0,30_a1.npy' \
      --tx2-file='movers_btil_svi_tx_human_FTTT_160_0,00_a2.npy'

# - flood (rescue_2)
python analysis/TIC_results/save_merged_v_values.py \
      --domain=rescue_2 --iteration=30 --output-suffix='fixedrobot' \
      --save-dir=human_data \
      --policy1-file='rescue_2_btil_dec_policy_human_woTx_FTTT_160_0,30_a1.npy' \
      --policy2-file='rescue_2_btil_svi_policy_human_woTx_FTTT_160_0,00_a2.npy' \
      --tx1-file='rescue_2_btil_dec_tx_human_FTTT_160_0,30_a1.npy' \
      --tx2-file='rescue_2_btil_svi_tx_human_FTTT_160_0,00_a2.npy'
