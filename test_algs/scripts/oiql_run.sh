#!/usr/bin/env bash

###### prerequisites: ######
# python3 >= 3.6
#     torch >= 1.7
#     tensorboard >= 2.2
#     matplotlib >= 3.2

########################## Option-IQL:
# python3 test_algs/run_algs.py --alg_name oiql --env_type mujoco \
#         --env_name Hopper-v2 --n_traj 1 --device "cuda:0" \
#         --bounded_actor True --tag oiqlstrm_256_3e-5_boundnnstd_extraD \
#         --use_prev_action False --data_path "experts/Hopper-v2_25.pkl" \
#         --max_explore_step 1e6 --mini_batch_size 256 --clip_grad_val 0 \
#         --use_nn_logstd True --clamp_action_logstd False --stream_training True \
#         --use_prev_action_dim True --use_prev_option_dim True \
#         --optimizer_lr_policy 3.e-5 --optimizer_lr_critic 3.e-4 --seed 0 \
#         --optimizer_lr_option 3.e-5 --separate_policy_update False \
#         --hidden_critic "(256, 256)" --demo_latent_infer_interval 5000 \
#         --init_temp 0.01 --n_sample 5000 --dim_c 4 --method_regularize True \
#         --method_loss v0


# python3 test_algs/run_algs.py --alg_name oiql --env_type mujoco \
#         --env_name CleanupSingle-v0 --n_traj 10 --device "cuda:0" \
#         --bounded_actor True --tag oiqlstrm_256_3e-5_boundnnstd_extraD \
#         --use_prev_action False --data_path "experts/CleanupSingle-v0_100.pkl" \
#         --max_explore_step 1e6 --mini_batch_size 256 --clip_grad_val 0 \
#         --use_nn_logstd True --clamp_action_logstd False --stream_training True \
#         --use_prev_action_dim True --use_prev_option_dim True \
#         --optimizer_lr_policy 3.e-5 --optimizer_lr_critic 3.e-4 --seed 0 \
#         --optimizer_lr_option 3.e-5 --separate_policy_update False \
#         --hidden_critic "(256, 256)" --demo_latent_infer_interval 5000 \
#         --init_temp 0.01 --n_sample 5000 --dim_c 4 --iql_single_critic True \
#         --supervision 0.0 --method_regularize True --method_loss value


python3 test_algs/run_algs.py --alg_name oiql \
        --env_name MultiGoals2D_2-v0 --n_traj 100 --device "cuda:0" \
        --tag oiql_256_3e-5_value --dim_c 2 --bounded_actor True \
        --data_path "experts/MultiGoals2D_2-v0_500.pkl" --max_explore_step 1e6 \
        --mini_batch_size 256 --n_sample 5000  --stream_training True \
        --demo_latent_infer_interval 3000 --n_update_rounds 500 \
        --use_prev_action False --max_explore_step 1e6 --clip_grad_val 0 \
        --use_nn_logstd True --clamp_action_logstd False \
        --use_prev_action_dim True --use_prev_option_dim True \
        --optimizer_lr_policy 3.e-5 --optimizer_lr_critic 3.e-4 --seed 0 \
        --optimizer_lr_option 3.e-5 --separate_policy_update False \
        --hidden_critic "(256, 256)" --init_temp 0.01 --iql_single_critic True \
        --supervision 0.0 --method_regularize True --method_loss value