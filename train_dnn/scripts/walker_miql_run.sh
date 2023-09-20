
########################### Mental-IQL:
# Walker2d-v2
python3 train_dnn/run_algs.py alg=miql env=Walker2d-v2 base=mujoco_base \
        tag=Ttx001Tpi001 miql_pi_method_loss=v0 \
        miql_tx_init_temp=1e-2 miql_pi_init_temp=1e-2

