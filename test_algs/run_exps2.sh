#!/usr/bin/env bash

###### prerequisites: ######
# python3 >= 3.6
#     torch >= 1.7
#     tensorboard >= 2.2
#     matplotlib >= 3.2

###### for Hopper-v2 #####
python3 test_algs/run_algs.py --alg_name miql_v2 --env_type mujoco \
        --env_name Hopper-v2 --n_demo 1000 --device "cuda:0" \
        --bounded_actor True --tag miqlv2-1k-a3c1 --use_prev_action False \
        --use_state_filter False --data_path "experts/Hopper-v2_25.pkl" \
        --num_actor_update 3 --num_critic_update 1