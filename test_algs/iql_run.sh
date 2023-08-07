#!/usr/bin/env bash

###### prerequisites: ######
# python3 >= 3.6
#     torch >= 1.7
#     tensorboard >= 2.2
#     matplotlib >= 3.2

###### for Hopper-v2 #####
# Mental-IQL:
python3 test_algs/run_algs.py --alg_name iql --env_type mujoco \
        --env_name Hopper-v2 --n_demo 1000 --device "cuda:0" \
        --bounded_actor True --tag iql_64_3e-5 \
        --use_prev_action False --data_path "experts/Hopper-v2_25.pkl" \
        --max_explore_step 3e6 --mini_batch_size 64 --use_nn_logstd True \
        --clip_grad_val 0 --clamp_action_logstd False \
        --use_prev_action_dim False --use_prev_option_dim False \
        --optimizer_lr_policy 3.e-5 --optimizer_lr_critic 3.e-5 --seed 0 \
        --learn_temp True --optimizer_lr_alpha 3.e-5

