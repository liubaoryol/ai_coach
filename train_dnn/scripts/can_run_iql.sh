# IQL
 
python train_dnn/run_algs.py alg=iql base=can_base \
       env=RMPickPlaceCan-v0 tag=N512x3LRhi init_temp=0.1 \
       hidden_critic=[512,512,512] hidden_policy=[512,512,512] \
       optimizer_lr_critic=1.e-3 optimizer_lr_policy=1e-4

