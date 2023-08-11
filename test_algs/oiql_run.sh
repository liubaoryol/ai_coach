#!/usr/bin/env bash

###### prerequisites: ######
# python3 >= 3.6
#     torch >= 1.7
#     tensorboard >= 2.2
#     matplotlib >= 3.2

###### for Hopper-v2 #####
# Option-IQL:
# python3 test_algs/run_algs.py --alg_name oiql --env_type mujoco \
#         --env_name Hopper-v2 --n_traj 1 --device "cuda:0" \
#         --bounded_actor True --tag oiqlstrm_64_3e-5_boundnnstd_extraD \
#         --use_prev_action False --data_path "experts/Hopper-v2_25.pkl" \
#         --max_explore_step 3e6 --mini_batch_size 64 --clip_grad_val 0 \
#         --use_nn_logstd True --clamp_action_logstd False --stream_training True \
#         --use_prev_action_dim True --use_prev_option_dim True \
#         --optimizer_lr_policy 3.e-5 --optimizer_lr_critic 3.e-5 --seed 0 \
#         --optimizer_lr_option 3.e-5 --separate_policy_update True \
#         --hidden_critic "(256, 256)" --demo_latent_infer_interval 5000 \
#         --init_temp 0.2

# python3 test_algs/run_algs.py --alg_name oiqlv2 --env_type mujoco \
#         --env_name Hopper-v2 --n_traj 1 --device "cuda:0" \
#         --bounded_actor True --tag oiqlv2_64_3e-5_boundnnstd \
#         --use_prev_action False --data_path "experts/Hopper-v2_25.pkl" \
#         --max_explore_step 3e6 --mini_batch_size 64 --clip_grad_val 0 \
#         --use_nn_logstd True --clamp_action_logstd False --stream_training True \
#         --use_prev_action_dim True --use_prev_option_dim True \
#         --optimizer_lr_policy 3.e-5 --optimizer_lr_critic 3.e-5 --seed 0 \
#         --optimizer_lr_option 3.e-5 --separate_policy_update True \
#         --hidden_critic "(256, 256)" --demo_latent_infer_interval 5000 \
#         --init_temp 0.2

# Mental-IQL:

python3 test_algs/run_algs.py --alg_name miql \
        --env_name Hopper-v2 --n_traj 1 --device "cpu" \
        --tag miql_256_3e-5 --seed 0 --dim_c 4 \
        --data_path "experts/Hopper-v2_25.pkl" --max_explore_step 1e6 \
        --mini_batch_size 256 --n_sample 5000  --stream_training True \
        --demo_latent_infer_interval 3000 --n_update_rounds 500 \
        --miql_update_strategy 1 --miql_tx_after_pi True \
        --miql_tx_method_loss "value" --miql_tx_method_regularize True \
        --miql_tx_optimizer_lr_critic 3.e-5 --miql_tx_init_temp 1e-2 \
        --miql_pi_optimizer_lr_critic 3.e-5 --miql_pi_optimizer_lr_alpha 3.e-5 \
        --miql_pi_optimizer_lr_policy 3.e-5 --miql_pi_init_temp 1e-2 \
        --miql_pi_learn_temp True --miql_pi_method_loss "value" \
        --miql_pi_method_regularize True \
        --miql_tx_hidden_critic "(64, 64)" --miql_pi_hidden_critic "(64, 64)" \
        --miql_pi_hidden_policy "(64, 64)"

python3 test_algs/run_algs.py --alg_name miql \
        --env_name Walker2d-v2 --n_traj 5 --device "cpu" \
        --tag miql_256_3e-5 --seed 0 --dim_c 4 \
        --data_path "experts/Walker2d-v2_25.pkl" --max_explore_step 1e6 \
        --mini_batch_size 256 --n_sample 5000  --stream_training True \
        --demo_latent_infer_interval 3000 --n_update_rounds 500 \
        --miql_update_strategy 1 --miql_tx_after_pi True \
        --miql_tx_method_loss "value" --miql_tx_method_regularize True \
        --miql_tx_optimizer_lr_critic 3.e-5 --miql_tx_init_temp 1e-2 \
        --miql_pi_optimizer_lr_critic 3.e-5 --miql_pi_optimizer_lr_alpha 3.e-5 \
        --miql_pi_optimizer_lr_policy 3.e-5 --miql_pi_init_temp 1e-2 \
        --miql_pi_learn_temp True --miql_pi_method_loss "value" \
        --miql_pi_method_regularize True \
        --miql_tx_hidden_critic "(64, 64)" --miql_pi_hidden_critic "(64, 64)" \
        --miql_pi_hidden_policy "(64, 64)"

# python3 test_algs/run_algs.py --alg_name miql \
#         --env_name MultiGoals2D_2-v0 --n_traj 300 --device "cpu" \
#         --tag miql_256_3e-5_value --seed 0 --dim_c 2 \
#         --data_path "experts/MultiGoals2D_2-v0_500.pkl" --max_explore_step 1e6 \
#         --mini_batch_size 256 --n_sample 5000  --stream_training True \
#         --demo_latent_infer_interval 3000 --n_update_rounds 500 \
#         --miql_update_strategy 1 --miql_tx_after_pi True \
#         --miql_tx_method_loss "value" --miql_tx_method_regularize True \
#         --miql_tx_optimizer_lr_critic 3.e-5 --miql_tx_init_temp 1e-2 \
#         --miql_pi_optimizer_lr_critic 3.e-5 --miql_pi_optimizer_lr_alpha 3.e-5 \
#         --miql_pi_optimizer_lr_policy 3.e-5 --miql_pi_init_temp 1e-2 \
#         --miql_pi_learn_temp True --miql_pi_method_loss "value" \
#         --miql_pi_method_regularize True \
#         --miql_tx_hidden_critic "(64, 64)" --miql_pi_hidden_critic "(64, 64)" \
#         --miql_pi_hidden_policy "(64, 64)"

# python3 test_algs/run_algs.py --alg_name miql \
#         --env_name MultiGoals2D_2-v0 --n_traj 300 --device "cpu" \
#         --tag miql_64_3e-5_pond --seed 0 --dim_c 2 \
#         --data_path "experts/MultiGoals2D_2-v0_500.pkl" --max_explore_step 3e6 \
#         --mini_batch_size 500 --n_sample 5000  --stream_training False \
#         --demo_latent_infer_interval 5000 --n_update_rounds 100 \
#         --miql_update_strategy 1 --miql_tx_after_pi True \
#         --miql_alter_update_n_pi_tx "(10, 5)" --miql_order_update_pi_ratio 0.5 \
#         --miql_tx_optimizer_lr_critic 3.e-5 --miql_tx_init_temp 1e-2 \
#         --miql_pi_optimizer_lr_critic 3.e-5 --miql_pi_optimizer_lr_alpha 3.e-5 \
#         --miql_pi_optimizer_lr_policy 3.e-5 --miql_pi_init_temp 1e-2 \
#         --miql_pi_learn_temp True --miql_pi_method_loss "value" \
#         --miql_tx_hidden_critic "(32, 32)" --miql_pi_hidden_critic "(32, 32)" \
#         --miql_pi_hidden_policy "(32, 32)"

# # LunarLander-v2
# python3 test_algs/run_algs.py --alg_name miql --env_type mujoco \
#         --env_name LunarLander-v2 --n_traj 10 --device "cuda:0" \
#         --tag miql_64_3e-5_test --seed 0 --dim_c 1 \
#         --data_path "experts/LunarLander-v2_1000.npy" \
#         --n_sample 5000  --stream_training True \
#         --demo_latent_infer_interval 5000 --n_update_rounds 500 \
#         --max_explore_step 3e6 --mini_batch_size 256 --clip_grad_val 0 \
#         --miql_update_strategy 1 --miql_tx_after_pi True \
#         --miql_tx_method_regularize True --miql_tx_method_loss "value" \
#         --miql_tx_optimizer_lr_critic 1.e-4 --miql_tx_init_temp 1e-2 \
#         --miql_pi_optimizer_lr_critic 1.e-4 --miql_pi_optimizer_lr_alpha 1.e-4 \
#         --miql_pi_optimizer_lr_policy 1.e-4 --miql_pi_init_temp 1e-2 \
#         --miql_pi_learn_temp True --miql_pi_method_loss "value" \
#         --miql_pi_method_regularize True \
#         --miql_tx_hidden_critic "(64, 64)" --miql_pi_hidden_critic "(64, 64)" \
#         --miql_pi_hidden_policy "(64, 64)"

# # CleanupSingle-v0
# python3 test_algs/run_algs.py --alg_name miql \
#         --env_name CleanupSingle-v0 --n_traj 10 --device "cpu" \
#         --tag miql_256_3e-5_value --seed 0 --dim_c 4 \
#         --data_path "experts/CleanupSingle-v0_100.pkl" --max_explore_step 1e6 \
#         --mini_batch_size 256 --n_sample 5000  --stream_training True \
#         --demo_latent_infer_interval 3000 --n_update_rounds 500 \
#         --miql_update_strategy 1 --miql_tx_after_pi True \
#         --miql_tx_method_loss "value" --miql_tx_method_regularize True \
#         --miql_tx_optimizer_lr_critic 3.e-5 --miql_tx_init_temp 1e-2 \
#         --miql_pi_optimizer_lr_critic 3.e-5 --miql_pi_optimizer_lr_alpha 3.e-5 \
#         --miql_pi_optimizer_lr_policy 3.e-5 --miql_pi_init_temp 1e-2 \
#         --miql_pi_learn_temp True --miql_pi_method_loss "value" \
#         --miql_pi_method_regularize True \
#         --miql_tx_hidden_critic "(64, 64)" --miql_pi_hidden_critic "(64, 64)" \
#         --miql_pi_hidden_policy "(64, 64)"