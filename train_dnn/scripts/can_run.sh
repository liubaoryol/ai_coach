# IQL
 
python train_dnn/run_algs.py alg=iql base=can_base \
       env=RMPickPlaceCan-v0 tag=N512x3LRhi

# MIQL

python train_dnn/run_algs.py alg=miql base=can_base \
       env=RMPickPlaceCan-v0 tag=Sv2_N256x3LRhiClip

# OIQL

python train_dnn/run_algs.py alg=oiql base=can_base \
       env=RMPickPlaceCan-v0 tag=Sv2_N512x3

# OGAIL

python train_dnn/run_algs.py alg=ogail base=can_base \
       env=RMPickPlaceCan-v0 tag=Sv2_N512x2_N256x2