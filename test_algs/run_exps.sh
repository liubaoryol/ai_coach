#!/usr/bin/env bash

###### prerequisites: ######
# python3 >= 3.6
#     torch >= 1.7
#     tensorboard >= 2.2
#     matplotlib >= 3.2

###### for Hopper-v2 #####
# H-BC:
# python3 test_algs/run_algs.py --alg_name bc --env_type mujoco \
#         --env_name Hopper-v2 --use_option True --loss_type MLE \
#         --n_demo 1000 --device "cuda:0" --use_state_filter False --tag hbc-1k \
#         --data_path "data/mujoco/Hopper-v2_sample.torch"

# python3 test_algs/run_algs.py --alg_name bc --env_type mujoco \
#         --env_name Hopper-v2 --use_option True --loss_type MLE \
#         --n_demo 1000 --device "cuda:0" --use_state_filter False --tag hbc-1k \
#         --data_path "experts/Hopper-v2_25.pkl"

# IQL:
# python3 test_algs/run_algs.py --alg_name iql --env_type mujoco \
#         --env_name Hopper-v2 --n_demo 1000 --device "cuda:0" \
#         --bounded_actor True --tag iql-1k --data_path "experts/Hopper-v2_25.pkl"

# Option-GAIL:
python3 test_algs/run_algs.py --alg_name gail --env_type mujoco \
        --env_name Hopper-v2 --use_option True --use_c_in_discriminator True \
        --n_demo 1000 --device "cuda:0" --use_state_filter False --tag option-gail-1k \
        --data_path "experts/Hopper-v2_25.pkl"

# Mental-IQL:
python3 test_algs/run_algs.py --alg_name miql --env_type mujoco \
        --env_name Hopper-v2 --n_demo 1000 --device "cuda:0" \
        --bounded_actor True --tag iql-1k-a1c1 --use_prev_action False \
        --data_path "experts/Hopper-v2_25.pkl" --num_actor_update 1 \
        --num_critic_update 1

python3 test_algs/run_algs.py --alg_name miql --env_type mujoco \
        --env_name Hopper-v2 --n_demo 1000 --device "cuda:0" \
        --bounded_actor True --tag iql-1k-a1c3 --use_prev_action False \
        --data_path "experts/Hopper-v2_25.pkl" --num_actor_update 1 \
        --num_critic_update 3

python3 test_algs/run_algs.py --alg_name miql --env_type mujoco \
        --env_name Hopper-v2 --n_demo 1000 --device "cuda:0" \
        --bounded_actor True --tag iql-1k-a3c1 --use_prev_action False \
        --data_path "experts/Hopper-v2_25.pkl" --num_actor_update 3 \
        --num_critic_update 1

# miql_v2
python3 test_algs/run_algs.py --alg_name miql_v2 --env_type mujoco \
        --env_name Hopper-v2 --n_demo 1000 --device "cuda:0" \
        --bounded_actor True --tag miqlv2-1k-a1c1 --use_prev_action False \
        --use_state_filter False --data_path "experts/Hopper-v2_25.pkl" \
        --num_actor_update 1 --num_critic_update 1

python3 test_algs/run_algs.py --alg_name miql_v2 --env_type mujoco \
        --env_name Hopper-v2 --n_demo 1000 --device "cuda:0" \
        --bounded_actor True --tag miqlv2-1k-a1c3 --use_prev_action False \
        --use_state_filter False --data_path "experts/Hopper-v2_25.pkl" \
        --num_actor_update 1 --num_critic_update 3

python3 test_algs/run_algs.py --alg_name miql_v2 --env_type mujoco \
        --env_name Hopper-v2 --n_demo 1000 --device "cuda:0" \
        --bounded_actor True --tag miqlv2-1k-a3c1 --use_prev_action False \
        --use_state_filter False --data_path "experts/Hopper-v2_25.pkl" \ 
        --num_actor_update 3 --num_critic_update 1