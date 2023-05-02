#!/usr/bin/env bash

###### prerequisites: ######
# python3 >= 3.6
#     torch >= 1.7
#     tensorboard >= 2.2
#     matplotlib >= 3.2

###### for Hopper-v2 #####
python3 test_algs/run_algs.py --alg_name gailv2 --env_type mujoco \
        --env_name Hopper-v2 --n_demo 1000 --device "cuda:0" \
        --bounded_actor True --tag gailv2-1k-bounded --use_prev_action False \
        --use_state_filter False --data_path "experts/Hopper-v2_25.pkl" \
        --max_explore_step 3e6 --clip_grad_val 0