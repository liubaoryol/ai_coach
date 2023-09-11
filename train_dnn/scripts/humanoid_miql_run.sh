
########################### Mental-IQL:
# Humanoid-v2
python3 train_dnn/run_algs.py alg=miql env=Humanoid-v2 base=mujoco_2x_base \
        tag=10tj2xnetTpi03 miql_pi_method_loss=v0 miql_pi_init_temp=0.3
