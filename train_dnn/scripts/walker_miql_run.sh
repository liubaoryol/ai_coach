
########################### Mental-IQL:
# Walker2d-v2
python3 train_dnn/run_algs.py alg=miql env=Walker2d-v2 base=walker_base \
        tag=txtemp1 miql_tx_init_temp=1

# python3 train_dnn/run_algs.py alg=miql env=Walker2d-v2 base=walker_base \
#         tag=txtemp0.1 miql_tx_init_temp=0.1

# python3 train_dnn/run_algs.py alg=miql env=Walker2d-v2 base=walker_base \
#         tag=txtemp1e-2 miql_tx_init_temp=1e-2

# python3 train_dnn/run_algs.py alg=miql env=Walker2d-v2 base=walker_base \
#         tag=txtemp1e-3 miql_tx_init_temp=1e-3

# python3 train_dnn/run_algs.py alg=miql env=Walker2d-v2 base=walker_base \
#         tag=txtemp10 miql_tx_init_temp=10

# # Humanoid-v2
# python3 train_dnn/run_algs.py alg=miql env=Humanoid-v2 base=walker_base \
#         tag=shortrun miql_pi_method_loss=v0 miql_pi_init_temp=0.3

# # Ant-v2
# python3 train_dnn/run_algs.py alg=miql env=Ant-v2 base=walker_base \
#         tag=shortrun miql_pi_method_loss=value miql_pi_init_temp=1e-3

# # HalfCheetah-v2
# python3 train_dnn/run_algs.py alg=miql env=HalfCheetah-v2 base=walker_base \
#         tag=shortrun miql_pi_method_loss=value miql_pi_init_temp=1e-3

