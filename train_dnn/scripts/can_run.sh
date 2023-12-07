# MIQL

python train_dnn/run_algs.py alg=miql base=can_base \
       env=RMPickPlaceCan-v0 tag=Sv2_256x2T001 supervision=0.2 \
       miql_tx_optimizer_lr_critic=3.e-4 miql_pi_optimizer_lr_critic=3.e-4 \
       miql_pi_optimizer_lr_policy=3.e-5 miql_pi_init_temp=0.01



# # OIQL

# python train_dnn/run_algs.py alg=oiql base=can_base \
#        env=RMPickPlaceCan-v0 tag=Sv2_N512x3

# # OGAIL

# python train_dnn/run_algs.py alg=ogail base=can_base \
#        env=RMPickPlaceCan-v0 tag=Sv2_N512x2_N256x2