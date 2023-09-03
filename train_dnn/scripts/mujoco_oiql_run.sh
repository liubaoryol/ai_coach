
########################### Mental-IQL:
# Hopper-v2
python3 train_dnn/run_algs.py alg=oiql env=Hopper-v2 base_setting=mujoco_base \
        tag=shortrun method_loss=v0 init_temp=1e-2
