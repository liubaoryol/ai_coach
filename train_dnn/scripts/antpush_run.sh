
# ########################### original
# # option-gail
# python3 train_dnn/run_algs.py env=AntPush-v0-original alg=ogail \
#         base=antpush_base tag=orig

# ########################### clipped
# # option-gail
# python3 train_dnn/run_algs.py env=AntPush-v0-clipped alg=ogail \
#         base=antpush_base tag=clip

# intent-iql
python3 train_dnn/run_algs.py env=AntPush-v0-clipped alg=miql \
        base=antpush_base tag=infer5k_pival demo_latent_infer_interval=5e3

# python3 train_dnn/run_algs.py env=AntPush-v0-clipped alg=miql \
#         base=antpush_base tag=infer1k_pival demo_latent_infer_interval=1e3

# python3 train_dnn/run_algs.py env=AntPush-v0-clipped alg=miql \
#         base=antpush_base tag=infer5k_piv0 miql_pi_method_loss=v0
