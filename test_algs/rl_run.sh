#!/usr/bin/env bash

###### prerequisites: ######
# python3 >= 3.6
#     torch >= 1.7
#     tensorboard >= 2.2
#     matplotlib >= 3.2

###### for Hopper-v2 #####
# python3 test_algs/run_algs.py --alg_name oppov2 --env_type mujoco \
#         --env_name AntPush-v0 --device "cuda:0" --use_prev_action False \
#         --bounded_actor True --tag oppov2_64_3e-4_bound_prmstd_256 \
#         --max_explore_step 6e6 --mini_batch_size 256 --clip_grad_val 0 \
#         --use_nn_logstd False --clamp_action_logstd True \
#         --use_prev_action_dim True --use_prev_option_dim True \
#         --optimizer_lr_policy 3.e-4 --optimizer_lr_critic 3.e-4 --seed 0 \
#         --optimizer_lr_option 3.e-4 --use_option True

# python3 test_algs/run_algs.py --alg_name osac --env_type mujoco \
#         --env_name Pendulum-v1 --device "cuda:0" --use_prev_action False \
#         --bounded_actor True --tag osac_64_3e-4_bound_nnlogstd_256 \
#         --max_explore_step 6e6 --mini_batch_size 64  --clip_grad_val 0 \
#         --use_nn_logstd True --clamp_action_logstd False --stream_training True \
#         --use_prev_action_dim True --use_prev_option_dim True \
#         --optimizer_lr_policy 3.e-4 --optimizer_lr_critic 3.e-4 --seed 0 \
#         --optimizer_lr_option 3.e-4 --separate_policy_update True \
#         --hidden_critic "(256, 256)" --learn_temp True

# python3 test_algs/run_algs.py --alg_name sb3_ppo --env_type mujoco \
#         --env_name Pendulum-v1 --device "cuda:0" \
#         --tag sb3ppo_64_3e-4_256 \
#         --max_explore_step 6e6 --mini_batch_size 64 --clip_grad_val 0 \
#         --optimizer_lr_policy 3.e-4 --optimizer_lr_critic 3.e-4 --seed 0 \
#         --optimizer_lr_option 3.e-4 --use_option FTrue