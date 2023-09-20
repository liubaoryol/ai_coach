########################### Option-IQL:
# Ant-v2
python3 train_dnn/run_algs.py alg=oiql env=Ant-v2 base=mujoco_base \
        tag=sep19 method_loss=value init_temp=1e-3

# Humanoid-v2
python3 train_dnn/run_algs.py alg=oiql env=Humanoid-v2 base=mujoco_base \
        tag=sep19 method_loss=v0 init_temp=0.3

# HalfCheetah-v2
python3 train_dnn/run_algs.py alg=oiql env=HalfCheetah-v2 base=mujoco_base \
        tag=sep19 method_loss=value init_temp=1e-2

# Hopper-v2
python3 train_dnn/run_algs.py alg=oiql env=Hopper-v2 base=hopper_base \
        tag=sep19 method_loss=v0 init_temp=1e-2

# Walker2d-v2
python3 train_dnn/run_algs.py alg=oiql env=Walker2d-v2 base=mujoco_base \
        tag=sep19 method_loss=v0 init_temp=1e-2

# AntPush-v0
python3 train_dnn/run_algs.py alg=oiql env=AntPush-v0-clipped base=antpush_base \
        tag=sep19 method_loss=value init_temp=1e-3
