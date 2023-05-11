#!/usr/bin/env bash

###### prerequisites: ######
# python3 >= 3.6
#     torch >= 1.7
#     tensorboard >= 2.2
#     matplotlib >= 3.2

###### for Hopper-v2 #####
#python3 test_algs/run_algs.py --alg_name oppo --env_type mujoco \
#        --env_name Hopper-v2 --device "cuda:0" --use_prev_action False \
#        --bounded_actor True --tag oppo_256_3e-4_nnlogstd_each64x64 \
#        --max_explore_step 6e6 --mini_batch_size 256  --clip_grad_val 0 \
#        --use_nn_logstd True --clamp_action_logstd False \
#        --use_prev_action_dim True --use_prev_option_dim True \
#        --optimizer_lr_policy 3.e-4 --optimizer_lr_critic 3.e-4 --seed 0 \
#        --optimizer_lr_option 3.e-4

python3 test_algs/run_algs.py --alg_name msac --env_type mujoco \
        --env_name Hopper-v2 --device "cuda:0" --use_prev_action False \
        --bounded_actor True --tag oppo_256_3e-4_prmlogstd_clamp_256x256 \
        --max_explore_step 6e6 --mini_batch_size 256  --clip_grad_val 0 \
        --use_nn_logstd False --clamp_action_logstd True \
        --use_prev_action_dim True --use_prev_option_dim True \
        --optimizer_lr_policy 3.e-4 --optimizer_lr_critic 3.e-4 --seed 0 \
        --optimizer_lr_option 3.e-4 --miql_stream False
