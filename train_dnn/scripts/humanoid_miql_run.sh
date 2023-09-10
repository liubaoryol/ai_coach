
########################### Mental-IQL:
# Humanoid-v2
python3 train_dnn/run_algs.py alg=miql env=Humanoid-v2 base=mujoco_base \
        tag=txT1_piT0.3 miql_pi_method_loss=v0 miql_pi_init_temp=0.3 \
        miql_tx_init_temp=1
