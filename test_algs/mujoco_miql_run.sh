
########################### Mental-IQL:
# Hopper-v2
python3 test_algs/run_mujoco.py --alg_name miql --env_type mujoco \
        --env_name Hopper-v2 --tag largenet --seed 0 \
        --data_path "experts/Hopper-v2_25.pkl" --n_sample 10000 \
        --miql_tx_method_loss "value"  --miql_pi_method_loss "v0" \
        --miql_pi_init_temp 1e-2

# # Walker2d-v2
# python3 test_algs/run_mujoco.py --alg_name miql --env_type mujoco \
#         --env_name Walker2d-v2 --tag largenet --seed 0 \
#         --data_path "experts/Walker2d-v2_25.pkl" --n_sample 10000 \
#         --miql_tx_method_loss "value"  --miql_pi_method_loss "v0" \
#         --miql_pi_init_temp 1e-2

# # Humanoid-v2
# python3 test_algs/run_mujoco.py --alg_name miql --env_type mujoco \
#         --env_name Humanoid-v2 --tag largenet --seed 0 \
#         --data_path "experts/Humanoid-v2_25.pkl" --n_sample 10000 \
#         --miql_tx_method_loss "value"  --miql_pi_method_loss "v0" \
#         --miql_pi_init_temp 0.3

# # Ant-v2
# python3 test_algs/run_mujoco.py --alg_name miql --env_type mujoco \
#         --env_name Ant-v2 --tag largenet --seed 0 \
#         --data_path "experts/Ant-v2_25.pkl" --n_sample 10000 \
#         --miql_tx_method_loss "value"  --miql_pi_method_loss "value" \
#         --miql_pi_init_temp 1e-3

# # HalfCheetah-v2
# python3 test_algs/run_mujoco.py --alg_name miql --env_type mujoco \
#         --env_name HalfCheetah-v2 --tag largenet --seed 0 \
#         --data_path "experts/HalfCheetah-v2_25.pkl" --n_sample 10000 \
#         --miql_tx_method_loss "value"  --miql_pi_method_loss "value" \
#         --miql_pi_init_temp 1e-2
