# IQL
 
python train_dnn/run_algs.py alg=miql base=can_base \
       env=RMPickPlaceCan-v0 tag=n2xSv2Ttx001Tpi03 \
       miql_tx_init_temp=0.01 miql_pi_init_temp=0.3

python train_dnn/run_algs.py alg=miql base=can_base \
       env=RMPickPlaceCan-v0 tag=n2xSv2Ttx001Tpi01 \
       miql_tx_init_temp=0.01 miql_pi_init_temp=0.1

python train_dnn/run_algs.py alg=miql base=can_base \
       env=RMPickPlaceCan-v0 tag=n2xSv2Ttx001Tpi003 \
       miql_tx_init_temp=0.01 miql_pi_init_temp=0.03

python train_dnn/run_algs.py alg=miql base=can_base \
       env=RMPickPlaceCan-v0 tag=n2xSv2Ttx001Tpi005 \
       miql_tx_init_temp=0.01 miql_pi_init_temp=0.05
