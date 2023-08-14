#!/usr/bin/env bash

###### prerequisites: ######
# python3 >= 3.6
#     torch >= 1.7
#     tensorboard >= 2.2
#     matplotlib >= 3.2

# option-gail
# # Hopper-v2
# python3 test_algs/run_algs.py --alg_name ogail --use_option True \
#         --env_name Hopper-v2 --n_traj 1 --device "cuda:0" \
#         --tag ogail_64_3e-5_value --seed 0 --dim_c 4 \
#         --data_path "experts/Hopper-v2_25.pkl" --max_explore_step 1e6 \
#         --mini_batch_size 64 --n_sample 5000 \
#         --use_c_in_discriminator True --use_state_filter False \
#         --log_std_bounds "(-20, 0)"

# CleanupSingle-v0
python3 test_algs/run_algs.py --alg_name ogail --use_option True \
        --env_name CleanupSingle-v0 --n_traj 10 --device "cuda:0" \
        --tag ogail_64_3e-5_value --seed 0 --dim_c 4 \
        --data_path "experts/CleanupSingle-v0_100.pkl" --max_explore_step 1e6 \
        --mini_batch_size 64 --n_sample 5000 \
        --use_c_in_discriminator True --use_state_filter False \
        --log_std_bounds "(-20, 0)"
