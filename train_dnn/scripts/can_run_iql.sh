# IQL
 
python train_dnn/run_algs.py alg=iql base=can_base \
       env=RMPickPlaceCan-v0 tag=T001v0 \
       method_loss=v0 init_temp=0.01

python train_dnn/run_algs.py alg=iql base=can_base \
       env=RMPickPlaceCan-v0 tag=T01v0 \
       method_loss=v0 init_temp=0.1

python train_dnn/run_algs.py alg=iql base=can_base \
       env=RMPickPlaceCan-v0 tag=T0005v0 \
       method_loss=v0 init_temp=0.005

python train_dnn/run_algs.py alg=iql base=can_base \
       env=RMPickPlaceCan-v0 tag=T005v0 \
       method_loss=v0 init_temp=0.05
