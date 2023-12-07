
########################### Mental-IQL:
# HalfCheetah-v2
python3 train_dnn/run_algs.py alg=miql env=HalfCheetah-v2 base=mujoco_base \
        tag=oct7seed2 seed=2 miql_pi_method_loss=value
