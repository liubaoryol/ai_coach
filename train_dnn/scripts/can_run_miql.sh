# IQL
 
python train_dnn/run_algs.py alg=miql base=can_base \
       env=RMPickPlaceCan-v0 tag=Sv2N256_V0_QdT01 miql_pi_init_temp=0.1 \
       miql_tx_method_loss=v0

python train_dnn/run_algs.py alg=miql base=can_base \
       env=RMPickPlaceCan-v0 tag=Sv2N256_V0_QdT008 miql_pi_init_temp=0.08 \
       miql_tx_method_loss=v0

python train_dnn/run_algs.py alg=miql base=can_base \
       env=RMPickPlaceCan-v0 tag=Sv2N256_Val_QdT008 miql_pi_init_temp=0.08 \
       miql_tx_method_loss=value

python train_dnn/run_algs.py alg=miql base=can_base \
       env=RMPickPlaceCan-v0 tag=Sv2N256_Val_QdT01 miql_pi_init_temp=0.1 \
       miql_tx_method_loss=value