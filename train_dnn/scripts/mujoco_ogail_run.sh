
########################### option-gail:
# Ant-v2
python3 train_dnn/run_algs.py alg=ogail env=Ant-v2 base=mujoco_base \
        tag=oct3seed1 seed=1

# Humanoid-v2
python3 train_dnn/run_algs.py alg=ogail env=Humanoid-v2 base=mujoco_base \
        tag=oct3seed1 seed=1

# HalfCheetah-v2
python3 train_dnn/run_algs.py alg=ogail env=HalfCheetah-v2 base=mujoco_base \
        tag=oct3seed1 seed=1

# Hopper-v2
python3 train_dnn/run_algs.py alg=ogail env=Hopper-v2 base=hopper_base \
        tag=oct3seed1 seed=1

# Walker2d-v2
python3 train_dnn/run_algs.py alg=ogail env=Walker2d-v2 base=mujoco_base \
        tag=oct3seed1 seed=1

# AntPush-v0
python3 train_dnn/run_algs.py alg=ogail env=AntPush-v0-original base=antpush_base \
        tag=oct3seed1 seed=1
