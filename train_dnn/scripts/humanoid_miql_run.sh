
########################### Mental-IQL:
# Humanoid-v2
python3 train_dnn/run_algs.py alg=miql env=Humanoid-v2 base=mujoco_base \
        tag=Ttx001Tpi03 miql_pi_method_loss=v0 \
        miql_tx_init_temp=1e-2 miql_pi_init_temp=0.3



