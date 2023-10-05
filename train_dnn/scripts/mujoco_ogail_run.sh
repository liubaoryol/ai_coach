
########################### option-gail:
# HalfCheetah-v2
python3 train_dnn/run_algs.py alg=ogail env=HalfCheetah-v2 base=mujoco_base \
        tag=oct3seed1 seed=1


# AntPush-v0
python3 train_dnn/run_algs.py alg=ogail env=AntPush-v0-original base=antpush_base \
        tag=oct3seed1 seed=1
