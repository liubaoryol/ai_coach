#!/usr/bin/env bash

###### prerequisites: ######
# python3 >= 3.6
#     torch >= 1.7
#     tensorboard >= 2.2
#     matplotlib >= 3.2

###### for Hopper-v2 #####
# Mental-IQL:
python3 test_algs/run_algs.py --alg_name miql --env_type mujoco \
        --env_name Hopper-v2 --n_demo 1000 --device "cuda:0" \
        --bounded_actor True --tag miqlpond_256_3e-5_prmlogstd_clamp_extra_dims \
        --use_prev_action False --data_path "experts/Hopper-v2_25.pkl" \
        --max_explore_step 3e6 --mini_batch_size 256 --use_nn_logstd False \
        --clip_grad_val 0 --clamp_action_logstd True \
        --use_prev_action_dim True --use_prev_option_dim True \
        --optimizer_lr_policy 3.e-5 --seed 0 \
        --demo_latent_infer_interval 5000 --miql_stream False

