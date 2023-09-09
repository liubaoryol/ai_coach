
########################### Mental-IQL:
# Hopper-v2
python3 train_dnn/run_algs.py alg=miql env=Hopper-v2 base=hopper_base \
        tag=piA3e-5 miql_pi_optimizer_lr_policy=3e-5 max_explore_step=1e6

python3 train_dnn/run_algs.py alg=miql env=Hopper-v2 base=hopper_base \
        tag=txtmp2e-2 miql_tx_init_temp=0.02 max_explore_step=1e6

# Humanoid-v2
python3 train_dnn/run_algs.py alg=miql env=Humanoid-v2 base=mujoco_base \
        tag=pitmp0.3 miql_pi_method_loss=v0 miql_pi_init_temp=0.3

python3 train_dnn/run_algs.py alg=miql env=Humanoid-v2 base=mujoco_base \
        tag=pitmp2e02 miql_pi_method_loss=v0 miql_pi_init_temp=2e-2

# Ant-v2
python3 train_dnn/run_algs.py alg=miql env=Ant-v2 base=mujoco_base \
        tag=pitmp1e-3 miql_pi_method_loss=value miql_pi_init_temp=1e-3

python3 train_dnn/run_algs.py alg=miql env=Ant-v2 base=mujoco_base \
        tag=pitmp2e-2 miql_pi_method_loss=value miql_pi_init_temp=2e-2

# HalfCheetah-v2
python3 train_dnn/run_algs.py alg=miql env=HalfCheetah-v2 base=mujoco_base \
        tag=txtmp2e-2 miql_pi_method_loss=value


# # Walker2d-v2
# python3 train_dnn/run_algs.py alg=miql env=Walker2d-v2 base=walker_base \
#         tag=test

