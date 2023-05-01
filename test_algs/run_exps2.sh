#!/usr/bin/env bash

###### prerequisites: ######
# python3 >= 3.6
#     torch >= 1.7
#     tensorboard >= 2.2
#     matplotlib >= 3.2

###### for Hopper-v2 #####
# Option-GAIL:
python3 test_algs/run_algs.py --alg_name gail --env_type mujoco \
        --env_name Hopper-v2 --use_option True --use_c_in_discriminator True \
        --n_demo 1000 --device "cuda:0" --use_state_filter False --tag option-gail-1k \
        --data_path "experts/Hopper-v2_25.pkl" --max_explore_step 3e6

