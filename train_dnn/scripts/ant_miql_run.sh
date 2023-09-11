
########################### Mental-IQL:
# Ant-v2
python3 train_dnn/run_algs.py alg=miql env=Ant-v2 base=mujoco_2x_base \
        tag=10tj2xnetTpi0001 miql_pi_method_loss=value miql_pi_init_temp=1e-3
