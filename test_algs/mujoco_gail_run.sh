
########################### option-gail:
# Hopper-v2
python3 test_algs/run_mujoco.py --alg_name ogail --env_type mujoco \
        --env_name Hopper-v2 --tag orig --seed 0 \
        --data_path "experts/Hopper-v2_25.pkl" --n_sample 4096 \
        --log_std_bounds "(-20, 0)"

# # Walker2d-v2
# python3 test_algs/run_mujoco.py --alg_name ogail --env_type mujoco \
#         --env_name Walker2d-v2 --tag orig --seed 0 \
#         --data_path "experts/Walker2d-v2_25.pkl" --n_sample 4096 \
#         --log_std_bounds "(-20, 0)"

# # Humanoid-v2
# python3 test_algs/run_mujoco.py --alg_name ogail --env_type mujoco \
#         --env_name Humanoid-v2 --tag orig --seed 0 \
#         --data_path "experts/Humanoid-v2_25.pkl" --n_sample 4096 \
#         --log_std_bounds "(-20, 0)"

# # Ant-v2
# python3 test_algs/run_mujoco.py --alg_name ogail --env_type mujoco \
#         --env_name Ant-v2 --tag orig --seed 0 \
#         --data_path "experts/Ant-v2_25.pkl" --n_sample 4096 \
#         --log_std_bounds "(-20, 0)"

# # HalfCheetah-v2
# python3 test_algs/run_mujoco.py --alg_name ogail --env_type mujoco \
#         --env_name HalfCheetah-v2 --tag orig --seed 0 \
#         --data_path "experts/HalfCheetah-v2_25.pkl" --n_sample 4096 \
#         --log_std_bounds "(-20, 0)"
