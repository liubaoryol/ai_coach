#!/usr/bin/env bash

###### prerequisites: ######
# python3 >= 3.6
#     torch >= 1.7
#     tensorboard >= 2.2
#     matplotlib >= 3.2

# IQL:
# MultiGoals2D_2-v0
# python3 test_algs/run_algs.py --alg_name iql --env_type mujoco \
#         --env_name MultiGoals2D_2-v0 --n_traj 300 --device "cuda:0" \
#         --bounded_actor True --tag iql_64_3e-5 \
#         --iql_agent_name "sac" --method_regularize True --method_loss "value" \
#         --data_path "experts/MultiGoals2D_2-v0_500.pkl" \
#         --max_explore_step 3e6 --mini_batch_size 256 --clip_grad_val 0 \
#         --optimizer_lr_policy 3.e-5 --optimizer_lr_critic 3.e-5 --seed 0 \
#         --learn_temp True --optimizer_lr_alpha 3.e-5 \
#         --init_temp 1e-2 \
#         --hidden_policy "(64, 64)" --hidden_critic "(64, 64)"

# # LunarLander-v2
# python3 test_algs/run_algs.py --alg_name iql --env_type mujoco \
#         --env_name LunarLander-v2 --n_traj 10 --device "cuda:0" \
#         --bounded_actor True --tag iql_64_3e-5 --n_sample 5000 \
#         --iql_agent_name "softq" --method_regularize True --method_loss "value" \
#         --data_path "experts/LunarLander-v2_1000.npy" \
#         --max_explore_step 3e6 --mini_batch_size 256 --clip_grad_val 0 \
#         --optimizer_lr_policy 1.e-4 --optimizer_lr_critic 1.e-4 --seed 0 \
#         --learn_temp True --optimizer_lr_alpha 1.e-4 \
#         --init_temp 1e-2 \
#         --hidden_policy "(64, 64)" --hidden_critic "(64, 64)"


# # CleanupSingle-v0
# python3 test_algs/run_algs.py --alg_name iql --env_type mujoco \
#         --env_name CleanupSingle-v0 --n_traj 100 --device "cpu" \
#         --bounded_actor True --tag iql_256_3e-5 --n_sample 5000 \
#         --iql_agent_name "softq" --method_regularize True --method_loss "value" \
#         --data_path "experts/CleanupSingle-v0_100.pkl" \
#         --max_explore_step 1e6 --mini_batch_size 256 --clip_grad_val 0 \
#         --optimizer_lr_policy 1.e-4 --optimizer_lr_critic 1.e-4 --seed 0 \
#         --learn_temp True --optimizer_lr_alpha 1.e-4 \
#         --init_temp 1e-2 \
#         --hidden_policy "(64, 64)" --hidden_critic "(64, 64)"

# Hopper-v2
# python3 test_algs/run_algs.py --alg_name iql --env_type mujoco \
#         --env_name Hopper-v2 --n_traj 1 --device "cuda:0" \
#         --bounded_actor True --tag iql_256_3e-5 --n_sample 5000 \
#         --iql_agent_name "sac" --method_regularize True --method_loss "v0" \
#         --data_path "experts/Hopper-v2_25.pkl" \
#         --max_explore_step 1e6 --mini_batch_size 256 --clip_grad_val 0 \
#         --optimizer_lr_policy 3.e-5 --optimizer_lr_critic 3.e-5 --seed 0 \
#         --learn_temp True --optimizer_lr_alpha 3.e-5 \
#         --init_temp 1e-2 \
#         --hidden_policy "(64, 64)" --hidden_critic "(64, 64)"

# Walker2d-v2
python3 test_algs/run_algs.py --alg_name iql --env_type mujoco \
        --env_name Walker2d-v2 --n_traj 5 --device "cpu" \
        --bounded_actor True --tag iql_256_3e-5 --n_sample 5000 \
        --iql_agent_name "sac" --method_regularize True --method_loss "v0" \
        --data_path "experts/Walker2d-v2_25.pkl" \
        --max_explore_step 1e6 --mini_batch_size 256 --clip_grad_val 0 \
        --optimizer_lr_policy 3.e-5 --optimizer_lr_critic 3.e-4 --seed 0 \
        --learn_temp True --optimizer_lr_alpha 3.e-4 \
        --init_temp 1e-2 \
        --hidden_policy "(256, 256)" --hidden_critic "(256, 256)"


# option-gail
# python3 test_algs/run_algs.py --alg_name ogail \
#         --env_name MultiGoals2D_2-v0 --n_traj 300 --device "cpu" \
#         --tag ogail_64_3e-5_value --seed 0 --dim_c 2 \
#         --data_path "experts/MultiGoals2D_2-v0_500.pkl" --max_explore_step 3e6 \
#         --use_option True \
#         --mini_batch_size 500 --n_sample 5000 
