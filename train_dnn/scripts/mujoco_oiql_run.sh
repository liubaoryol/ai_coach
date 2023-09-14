
########################### Mental-IQL:
# Hopper-v2
python3 train_dnn/run_algs.py alg=oiql env=Hopper-v2 base=mujoco_base \
        tag=10tjv0T001 method_loss=v0 init_temp=1e-2 n_traj=10

# # Walker2d-v2
# python3 train_dnn/run_algs.py alg=oiql env=Walker2d-v2 base=mujoco_base \
#         tag=10tjv0T001 method_loss=v0 init_temp=1e-2 n_traj=10

# # Ant-v2
# python3 train_dnn/run_algs.py alg=oiql env=Ant-v2 base=mujoco_base \
#         tag=10tjvalT0001 method_loss=val init_temp=1e-3 n_traj=10

# # HalfCheetah-v2
# python3 train_dnn/run_algs.py alg=oiql env=HalfCheetah-v2 base=mujoco_base \
#         tag=10tjvalT001 method_loss=val init_temp=1e-2 n_traj=10

# # Humanoid-v2
# python3 train_dnn/run_algs.py alg=oiql env=Humanoid-v2 base=mujoco_base \
#         tag=10tjv0T03 method_loss=v0 init_temp=0.3 n_traj=10
