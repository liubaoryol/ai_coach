
########################### Option-IQL:
# HalfCheetah-v2
#python3 train_dnn/run_algs.py alg=oiql env=HalfCheetah-v2 base=mujoco_base \
        #tag=oct3seed2 seed=2 method_loss=value init_temp=1e-2


# # AntPush-v0
python3 train_dnn/run_algs.py alg=oiql env=AntPush-v0-clipped base=antpush_base \
        tag=oct3seed2 seed=2 method_loss=value init_temp=1e-3
