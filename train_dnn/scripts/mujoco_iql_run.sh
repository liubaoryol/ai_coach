
########################### Mental-IQL:
# Ant-v2
python3 train_dnn/run_algs.py alg=iql env=Ant-v2 base=mujoco_base \
        tag=rem50kskip5val method_loss=value init_temp=1e-3

python3 train_dnn/run_algs.py alg=iql env=Ant-v2 base=mujoco_base \
        tag=rem50kskip5v0 method_loss=v0 init_temp=1e-3

# Humanoid-v2
python3 train_dnn/run_algs.py alg=iql env=Humanoid-v2 base=mujoco_base \
        tag=rem50kskip5v0 method_loss=v0 init_temp=1

python3 train_dnn/run_algs.py alg=iql env=Humanoid-v2 base=mujoco_base \
        tag=rem50kskip5val method_loss=value init_temp=1
