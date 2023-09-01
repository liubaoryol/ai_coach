
########################### Mental-IQL:
# Hopper-v2
# python3 test_algs/run_algs.py --alg_name miql \
#         --env_name Hopper-v2 --n_traj 1 --device "cpu" \
#         --tag miql_256_3e-5 --seed 0 --dim_c 4 \
#         --data_path "experts/Hopper-v2_25.pkl" --max_explore_step 1e6 \
#         --mini_batch_size 256 --n_sample 5000  --stream_training True \
#         --demo_latent_infer_interval 5000 --n_update_rounds 500 \
#         --miql_update_strategy 1 --miql_tx_after_pi True \
#         --miql_tx_method_loss "value" --miql_tx_method_regularize True \
#         --miql_tx_optimizer_lr_critic 3.e-5 --miql_tx_init_temp 1e-2 \
#         --miql_pi_optimizer_lr_critic 3.e-5 --miql_pi_optimizer_lr_alpha 3.e-5 \
#         --miql_pi_optimizer_lr_policy 3.e-5 --miql_pi_init_temp 1e-2 \
#         --miql_pi_learn_temp True --miql_pi_method_loss "v0" \
#         --miql_pi_method_regularize True --miql_pi_single_critic True \
#         --miql_tx_hidden_critic "(64, 64)" --miql_pi_hidden_critic "(64, 64)" \
#         --miql_pi_hidden_policy "(64, 64)"

# # Walker2d-v2
# python3 test_algs/run_algs.py --alg_name miql \
#         --env_name Walker2d-v2 --n_traj 5 --device "cuda:0" \
#         --tag miql_1Q_notemp_m10k_txregval --seed 0 --dim_c 4 \
#         --data_path "experts/Walker2d-v2_25.pkl" --max_explore_step 1e6 \
#         --mini_batch_size 256 --n_sample 10000  --stream_training True \
#         --demo_latent_infer_interval 5000 --n_update_rounds 500 \
#         --miql_update_strategy 1 --miql_tx_after_pi True \
#         --miql_tx_method_loss "value" --miql_tx_method_regularize True \
#         --miql_tx_optimizer_lr_critic 1.e-4 --miql_tx_init_temp 1e-2 \
#         --miql_pi_optimizer_lr_critic 3.e-4 --miql_pi_optimizer_lr_alpha 3.e-4 \
#         --miql_pi_optimizer_lr_policy 3.e-5 --miql_pi_init_temp 1e-2 \
#         --miql_pi_learn_temp False --miql_pi_method_loss "v0" \
#         --miql_pi_method_regularize True --miql_pi_single_critic True \
#         --miql_tx_hidden_critic "(32, 32)" --miql_pi_hidden_critic "(64, 64)" \
#         --miql_pi_hidden_policy "(64, 64)"

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

# EnvMovers-v0
python3 test_algs/run_algs.py --alg_name miql \
        --env_name EnvMovers-v0 --n_traj 44 --device "cuda:0" \
        --tag miql_256_3e-5_value --seed 0 --dim_c 5 \
        --data_path "experts/EnvMovers_v0_44.pkl" --max_explore_step 1e6 \
        --mini_batch_size 256 --n_sample 2000  --stream_training True \
        --demo_latent_infer_interval 2000 --n_update_rounds 500 \
        --miql_update_strategy 1 --miql_tx_after_pi True \
        --miql_tx_method_loss "value" --miql_tx_method_regularize True \
        --miql_tx_optimizer_lr_critic 3.e-5 --miql_tx_init_temp 1e-2 \
        --miql_pi_optimizer_lr_critic 3.e-5 --miql_pi_optimizer_lr_alpha 3.e-5 \
        --miql_pi_optimizer_lr_policy 3.e-5 --miql_pi_init_temp 1e-2 \
        --miql_pi_learn_temp False --miql_pi_method_loss "value" \
        --miql_pi_method_regularize True \
        --miql_tx_hidden_critic "(16, 16)" --miql_pi_hidden_critic "(32, 32)" \
        --miql_pi_hidden_policy "(32, 32)" --supervision 1.0

# # EnvCleanup_v0-v0
# python3 test_algs/run_algs.py --alg_name miql \
#         --env_name EnvCleanup-v0 --n_traj 44 --device "cuda:0" \
#         --tag miql_256_3e-5_value --seed 0 --dim_c 5 \
#         --data_path "experts/EnvCleanup_v0_66.pkl" --max_explore_step 1e6 \
#         --mini_batch_size 256 --n_sample 2000  --stream_training True \
#         --demo_latent_infer_interval 2000 --n_update_rounds 500 \
#         --miql_update_strategy 1 --miql_tx_after_pi True \
#         --miql_tx_method_loss "value" --miql_tx_method_regularize True \
#         --miql_tx_optimizer_lr_critic 3.e-5 --miql_tx_init_temp 1e-2 \
#         --miql_pi_optimizer_lr_critic 3.e-5 --miql_pi_optimizer_lr_alpha 3.e-5 \
#         --miql_pi_optimizer_lr_policy 3.e-5 --miql_pi_init_temp 1e-2 \
#         --miql_pi_learn_temp False --miql_pi_method_loss "value" \
#         --miql_pi_method_regularize True \
#         --miql_tx_hidden_critic "(32, 32)" --miql_pi_hidden_critic "(64, 64)" \
#         --miql_pi_hidden_policy "(64, 64)" --supervision 0.0
