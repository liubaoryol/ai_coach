
# IQL
# MultiGoals2D_2-v0
python3 train_dnn/run_algs.py alg=iql base_setting=MultiGoals2D_base \
        env=MultiGoals2D_2-v0,MultiGoals2D_3-v0,MultiGoals2D_4-v0,MultiGoals2D_5-v0 \
        n_traj=50 tag=testrun

# # CleanupSingle-v0
# python3 test_algs/run_simple2d.py --n_traj 50 --dim_c 4 \
#         --env_name CleanupSingle-v0 --data_path "experts/CleanupSingle-v0_100.pkl" \
#         --alg_name iql --n_sample 2000 --use_nn_logstd True --clamp_action_logstd False \
#         --iql_agent_name "softq"

# # EnvMovers-v0
# python3 test_algs/run_simple2d.py --n_traj 44 --dim_c 5 \
#         --env_name EnvMovers-v0 --data_path "experts/EnvMovers_v0_44.pkl" \
#         --alg_name iql --n_sample 2000 --use_nn_logstd True --clamp_action_logstd False \
#         --iql_agent_name "softq"

# # EnvCleanup-v0
# python3 test_algs/run_simple2d.py --n_traj 66 --dim_c 5 \
#         --env_name EnvCleanup-v0 --data_path "experts/EnvCleanup_v0_66.pkl" \
#         --alg_name iql --n_sample 2000 --use_nn_logstd True --clamp_action_logstd False \
#         --iql_agent_name "softq"