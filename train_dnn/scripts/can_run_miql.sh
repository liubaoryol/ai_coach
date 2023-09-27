# MIQL
 
# python train_dnn/run_algs.py alg=miql base=can_base \
#        env=RMPickPlaceCan-v0 tag=Sv2N256_Val_QdT01 miql_pi_init_temp=0.1 \
#        miql_tx_method_loss=value

python train_dnn/run_algs.py alg=miql base=can_base \
       env=RMPickPlaceCan-v0 tag=Sv2_N512x3LRhi miql_pi_init_temp=0.1 \
       miql_tx_method_loss=value supervision=0.2 miql_pi_single_critic=False \
       miql_pi_hidden_critic=[256,256,256] miql_pi_hidden_policy=[256,256,256] \
       miql_tx_optimizer_lr_critic=1e-3 miql_pi_optimizer_lr_critic=1e-3 \
       miql_pi_optimizer_lr_policy=1e-4
