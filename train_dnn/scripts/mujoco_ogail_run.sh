
########################### option-gail:
# Ant-v2
python3 train_dnn/run_algs.py alg=ogail env=Ant-v2 base=mujoco_base \
        tag=sep19

# Humanoid-v2
python3 train_dnn/run_algs.py alg=ogail env=Humanoid-v2 base=mujoco_base \
        tag=sep19

# HalfCheetah-v2
python3 train_dnn/run_algs.py alg=ogail env=HalfCheetah-v2 base=mujoco_base \
        tag=sep19

# Hopper-v2
python3 train_dnn/run_algs.py alg=ogail env=Hopper-v2 base=hopper_base \
        tag=sep19

# Walker2d-v2
python3 train_dnn/run_algs.py alg=ogail env=Walker2d-v2 base=mujoco_base \
        tag=sep19

# AntPush-v0
python3 train_dnn/run_algs.py alg=ogail env=AntPush-v0-original base=antpush_base \
        tag=sep19
