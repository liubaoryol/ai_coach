#!/usr/bin/env bash

###### prerequisites: ######
# python3 >= 3.6
#     torch >= 1.7
#     tensorboard >= 2.2
#     matplotlib >= 3.2

###### for Hopper-v2 #####
# Option-IQL:
# python3 test_algs/run_algs.py --alg_name oiql --env_type mujoco \
#         --env_name Hopper-v2 --n_demo 1000 --device "cuda:0" \
#         --bounded_actor True --tag oiqlstrm_64_3e-5_boundnnstd_extraD \
#         --use_prev_action False --data_path "experts/Hopper-v2_25.pkl" \
#         --max_explore_step 3e6 --mini_batch_size 64 --clip_grad_val 0 \
#         --use_nn_logstd True --clamp_action_logstd False --oiql_stream True \
#         --use_prev_action_dim True --use_prev_option_dim True \
#         --optimizer_lr_policy 3.e-5 --optimizer_lr_critic 3.e-5 --seed 0 \
#         --optimizer_lr_option 3.e-5 --separate_policy_update True \
#         --hidden_critic "(256, 256)" --demo_latent_infer_interval 5000 \
#         --init_temp 0.2

# python3 test_algs/run_algs.py --alg_name oiqlv2 --env_type mujoco \
#         --env_name Hopper-v2 --n_demo 1000 --device "cuda:0" \
#         --bounded_actor True --tag oiqlv2_64_3e-5_boundnnstd \
#         --use_prev_action False --data_path "experts/Hopper-v2_25.pkl" \
#         --max_explore_step 3e6 --mini_batch_size 64 --clip_grad_val 0 \
#         --use_nn_logstd True --clamp_action_logstd False --oiql_stream True \
#         --use_prev_action_dim True --use_prev_option_dim True \
#         --optimizer_lr_policy 3.e-5 --optimizer_lr_critic 3.e-5 --seed 0 \
#         --optimizer_lr_option 3.e-5 --separate_policy_update True \
#         --hidden_critic "(256, 256)" --demo_latent_infer_interval 5000 \
#         --init_temp 0.2

# Mental-IQL:
python3 test_algs/run_algs.py --alg_name miql --env_type mujoco \
        --env_name Hopper-v2 --n_demo 1000 --device "cuda:0" \
        --tag miql_64_tx3e-4_pi3-5_boundnnstd --seed 0 --dim_c 2 \
        --data_path "experts/Hopper-v2_25.pkl" --max_explore_step 3e6 \
        --mini_batch_size 64 --demo_latent_infer_interval 5000 \
        --miql_tx_optimizer_lr_critic 3.e-4 --miql_tx_init_temp 1e-2 \
        --miql_pi_optimizer_lr_critic 3.e-5 --miql_pi_optimizer_lr_alpha 3.e-5 \
        --miql_pi_optimizer_lr_policy 3.e-5 --miql_pi_init_temp 1e-2 \
        --miql_pi_learn_temp True
