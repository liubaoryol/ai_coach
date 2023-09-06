
########################### Mental-IQL:
# Hopper-v2
python3 train_dnn/run_algs.py alg=miql env=Hopper-v2 base=hopper_base \
        tag=txlr1e-4 miql_tx_optimizer_lr_critic=1e-4

python3 train_dnn/run_algs.py alg=miql env=Hopper-v2 base=hopper_base \
        tag=txlr3e-4 miql_tx_optimizer_lr_critic=3e-4

python3 train_dnn/run_algs.py alg=miql env=Hopper-v2 base=hopper_base \
        tag=txlr1e-5 miql_tx_optimizer_lr_critic=1e-5

python3 train_dnn/run_algs.py alg=miql env=Hopper-v2 base=hopper_base \
        tag=txlr3e-5 miql_tx_optimizer_lr_critic=3e-5


# # Walker2d-v2
# python3 train_dnn/run_algs.py alg=miql env=Walker2d-v2 base=hopper_base \
#         tag=shortrun miql_pi_method_loss=v0 miql_pi_init_temp=1e-2

# # Humanoid-v2
# python3 train_dnn/run_algs.py alg=miql env=Humanoid-v2 base=hopper_base \
#         tag=shortrun miql_pi_method_loss=v0 miql_pi_init_temp=0.3

# # Ant-v2
# python3 train_dnn/run_algs.py alg=miql env=Ant-v2 base=hopper_base \
#         tag=shortrun miql_pi_method_loss=value miql_pi_init_temp=1e-3

# # HalfCheetah-v2
# python3 train_dnn/run_algs.py alg=miql env=HalfCheetah-v2 base=hopper_base \
#         tag=shortrun miql_pi_method_loss=value miql_pi_init_temp=1e-3

