
# ########################### original
# # option-gail
# python3 train_dnn/run_algs.py env=AntPush-v0-original alg=ogail \
#         base=antpush_base tag=orig

# ########################### clipped
# # option-gail
# python3 train_dnn/run_algs.py env=AntPush-v0-clipped alg=ogail \
#         base=antpush_base tag=clip

# intent-iql
# python3 train_dnn/run_algs.py env=AntPush-v0-clipped alg=miql \
#         base=antpush_base tag=infer1k_pival demo_latent_infer_interval=1e3

# oiql
python train_dnn/run_algs.py env=AntPush-v0-clipped alg=oiql \
       base=antpush_base tag=val demo_latent_infer_interval=1e3 method_loss=v0 \
       init_temp=1e-3