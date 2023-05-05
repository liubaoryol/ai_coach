#!/usr/bin/env bash

###### prerequisites: ######
# python3 >= 3.6
#     torch >= 1.7
#     tensorboard >= 2.2
#     matplotlib >= 3.2

###### for Hopper-v2 #####
# IQL:
# python3 test_algs/run_algs.py --alg_name iql --env_type mujoco \
#         --env_name Hopper-v2 --n_demo 1000 --device "cuda:0" \
#         --bounded_actor True --tag iql-test --iql_agent_name sac\
#         --data_path "experts/Hopper-v2_25.pkl"

# Option-GAIL:
# python3 test_algs/run_algs.py --alg_name gail --env_type mujoco \
#         --env_name Hopper-v2 --use_option True --use_c_in_discriminator True \
#         --n_demo 1000 --device "cuda:0" --use_state_filter False --tag option-gail-test \
#         --data_path "experts/Hopper-v2_25.pkl"

# Option-GAIL v2:
# python3 test_algs/run_algs.py --alg_name gailv2 --env_type mujoco \
#         --env_name Hopper-v2 --use_option True --use_c_in_discriminator True \
#         --n_demo 1000 --device "cuda:0" --use_state_filter False --tag option-gailv2-test \
#         --data_path "experts/Hopper-v2_25.pkl" --bounded_actor False

# Mental-IQL:
# python3 test_algs/run_algs.py --alg_name miql --env_type mujoco \
#         --env_name Hopper-v2 --n_demo 1000 --device "cuda:0" \
#         --bounded_actor True --tag miql-test --use_prev_action False \
#         --data_path "experts/Hopper-v2_25.pkl" 

# miql_v2
python3 test_algs/run_algs.py --alg_name miqlv2 --env_type mujoco \
        --env_name Hopper-v2 --n_demo 1000 --device "cuda:0" \
        --bounded_actor True --tag miqlv2-test --use_prev_action False \
        --use_state_filter False --data_path "experts/Hopper-v2_25.pkl" 
