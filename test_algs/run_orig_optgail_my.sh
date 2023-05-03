#!/usr/bin/env bash

###### prerequisites: ######
# python3 >= 3.6
#     torch >= 1.7
#     tensorboard >= 2.2
#     matplotlib >= 3.2

###### for Hopper-v2 #####
python3 test_algs/run_algs.py --alg_name gailv2 --env_type mujoco \
        --env_name Hopper-v2 --n_demo 1000 --device "cuda:0" \
        --bounded_actor False --tag my_ogail_3e4_log_opt --use_prev_action False \
        --use_state_filter False --data_path "experts/Hopper-v2_25.pkl" \
        --max_explore_step 3e6 --clip_grad_val 0 \
        --gail_option_entropy_orig True --gail_action_sample_orig True \
        --gail_option_sample_orig True --gail_orig_log_opt False
