
########################### Mental-IQL:
# Ant-v2
python3 train_dnn/run_algs.py alg=miql env=Ant-v2 base=mujoco_base \
        tag=Ttx001Tpi0001 miql_pi_method_loss=value \
        miql_tx_init_temp=1e-2 miql_pi_init_temp=1e-3

