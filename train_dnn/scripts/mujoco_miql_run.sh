
########################### Mental-IQL:
# Hopper-v2
python3 train_dnn/run_algs.py alg=miql env=Hopper-v2 base_setting=mujoco_base \
        tag=txtemp0.01 miql_pi_method_loss=v0 miql_pi_init_temp=0.02 miql_tx_init_temp=0.01

python3 train_dnn/run_algs.py alg=miql env=Hopper-v2 base_setting=mujoco_base \
        tag=txtemp0.005 miql_pi_method_loss=v0 miql_pi_init_temp=0.02 miql_tx_init_temp=0.005

python3 train_dnn/run_algs.py alg=miql env=Hopper-v2 base_setting=mujoco_base \
        tag=txtemp0.1 miql_pi_method_loss=v0 miql_pi_init_temp=0.02 miql_tx_init_temp=0.1

python3 train_dnn/run_algs.py alg=miql env=Hopper-v2 base_setting=mujoco_base \
        tag=txtemp0.02 miql_pi_method_loss=v0 miql_pi_init_temp=0.02 miql_tx_init_temp=0.02


# # Walker2d-v2
# python3 train_dnn/run_algs.py alg=miql env=Walker2d-v2 base_setting=mujoco_base \
#         tag=shortrun miql_pi_method_loss=v0 miql_pi_init_temp=1e-2

# # Humanoid-v2
# python3 train_dnn/run_algs.py alg=miql env=Humanoid-v2 base_setting=mujoco_base \
#         tag=shortrun miql_pi_method_loss=v0 miql_pi_init_temp=0.3

# # Ant-v2
# python3 train_dnn/run_algs.py alg=miql env=Ant-v2 base_setting=mujoco_base \
#         tag=shortrun miql_pi_method_loss=value miql_pi_init_temp=1e-3

# # HalfCheetah-v2
# python3 train_dnn/run_algs.py alg=miql env=HalfCheetah-v2 base_setting=mujoco_base \
#         tag=shortrun miql_pi_method_loss=value miql_pi_init_temp=1e-3

