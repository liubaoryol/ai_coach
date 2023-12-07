# IQL

# python train_dnn/run_algs.py alg=iql base=can_base \
#        env=RMPickPlaceCan-v0 max_explore_step=1e7 tag=base512x3 \
#        n_sample=5e4 update_interval=5 mini_batch_size=256 \
#        hidden_critic=[512,512,512] hidden_policy=[512,512,512] \
#        optimizer_lr_critic=1.e-3 optimizer_lr_policy=1.e-4 \
#        method_loss=v0 init_temp=0.1 iql_single_critic=False
 
# python train_dnn/run_algs.py alg=iql base=can_base \
#        env=RMPickPlaceCan-v0 max_explore_step=1e7 tag=512x3T001 \
#        n_sample=5e4 update_interval=5 mini_batch_size=256 \
#        hidden_critic=[512,512,512] hidden_policy=[512,512,512] \
#        optimizer_lr_critic=1.e-3 optimizer_lr_policy=1.e-4 \
#        method_loss=v0 init_temp=0.01 iql_single_critic=False

python train_dnn/run_algs.py alg=iql base=can_base \
       env=RMPickPlaceCan-v0 max_explore_step=1e7 tag=512x2ValT001 \
       n_sample=5e4 update_interval=5 mini_batch_size=256 \
       hidden_critic=[512,512] hidden_policy=[512,512] \
       optimizer_lr_critic=3.e-4 optimizer_lr_policy=3.e-5 \
       method_loss=value init_temp=0.01 iql_single_critic=False
