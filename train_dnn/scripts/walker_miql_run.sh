
########################### Mental-IQL:
# Walker2d-v2
python3 train_dnn/run_algs.py alg=miql env=Walker2d-v2 base=mujoco_base \
        tag=pitemplearn miql_pi_method_loss=v0 miql_pi_learn_temp=True

python3 train_dnn/run_algs.py alg=miql env=Walker2d-v2 base=mujoco_base \
        tag=pitemp0.01 miql_pi_method_loss=v0 miql_pi_init_temp=0.01

python3 train_dnn/run_algs.py alg=miql env=Walker2d-v2 base=mujoco_base \
        tag=pitemp0.001 miql_pi_method_loss=v0 miql_pi_init_temp=0.001

python3 train_dnn/run_algs.py alg=miql env=Walker2d-v2 base=mujoco_base \
        tag=pitemp0.1 miql_pi_method_loss=v0 miql_pi_init_temp=0.1

# python3 train_dnn/run_algs.py alg=miql env=Walker2d-v2 base=mujoco_base \
#         tag=pitemp0.05 miql_pi_method_loss=v0 miql_pi_init_temp=0.05

# python3 train_dnn/run_algs.py alg=miql env=Walker2d-v2 base=mujoco_base \
#         tag=pitemp0.5 miql_pi_method_loss=v0 miql_pi_init_temp=0.5

# # Humanoid-v2
# python3 train_dnn/run_algs.py alg=miql env=Humanoid-v2 base=mujoco_base \
#         tag=shortrun miql_pi_method_loss=v0 miql_pi_init_temp=0.3

# # Ant-v2
# python3 train_dnn/run_algs.py alg=miql env=Ant-v2 base=mujoco_base \
#         tag=shortrun miql_pi_method_loss=value miql_pi_init_temp=1e-3

# # HalfCheetah-v2
# python3 train_dnn/run_algs.py alg=miql env=HalfCheetah-v2 base=mujoco_base \
#         tag=shortrun miql_pi_method_loss=value miql_pi_init_temp=1e-3

