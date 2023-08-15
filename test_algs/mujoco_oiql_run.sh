
########################### Mental-IQL:
# Hopper-v2
python3 test_algs/run_mujoco.py --alg_name oiql --env_type mujoco \
        --env_name Hopper-v2 --tag single_q --seed 0 \
        --data_path "experts/Hopper-v2_25.pkl" --n_sample 10000 \
        --optimizer_lr_critic 3e-4 --optimizer_lr_policy 3e-5 \
        --optimizer_lr_option 3e-5 --clamp_action_logstd False \
        --use_nn_logstd True --method_loss "v0"

# # Walker2d-v2
# python3 test_algs/run_mujoco.py --alg_name oiql --env_type mujoco \
#         --env_name Walker2d-v2 --tag single_q --seed 0 \
#         --data_path "experts/Walker2d-v2_25.pkl" --n_sample 10000 \
#         --optimizer_lr_critic 3e-4 --optimizer_lr_policy 3e-5 \
#         --optimizer_lr_option 3e-5 --clamp_action_logstd False \
#         --use_nn_logstd True --method_loss "v0"

# # Humanoid-v2
# python3 test_algs/run_mujoco.py --alg_name oiql --env_type mujoco \
#         --env_name Humanoid-v2 --tag single_q --seed 0 \
#         --data_path "experts/Humanoid-v2_25.pkl" --n_sample 10000 \
#         --optimizer_lr_critic 3e-4 --optimizer_lr_policy 3e-5 \
#         --optimizer_lr_option 3e-5 --clamp_action_logstd False \
#         --use_nn_logstd True --method_loss "v0"

# # Ant-v2
# python3 test_algs/run_mujoco.py --alg_name oiql --env_type mujoco \
#         --env_name Ant-v2 --tag single_q --seed 0 \
#         --data_path "experts/Ant-v2_25.pkl" --n_sample 10000 \
#         --optimizer_lr_critic 3e-4 --optimizer_lr_policy 3e-5 \
#         --optimizer_lr_option 3e-5 --clamp_action_logstd False \
#         --use_nn_logstd True --method_loss "value"

# # HalfCheetah-v2
# python3 test_algs/run_mujoco.py --alg_name oiql --env_type mujoco \
#         --env_name HalfCheetah-v2 --tag single_q --seed 0 \
#         --data_path "experts/HalfCheetah-v2_25.pkl" --n_sample 10000 \
#         --optimizer_lr_critic 3e-4 --optimizer_lr_policy 3e-5 \
#         --optimizer_lr_option 3e-5 --clamp_action_logstd False \
#         --use_nn_logstd True --method_loss "value"
