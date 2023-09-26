# IQL
 
python train_dnn/run_algs.py alg=iql base=can_base \
       env=RMPickPlaceCan-v0 tag=n2xT0010 \
       method_loss=v0 init_temp=0.01

python train_dnn/run_algs.py alg=iql base=can_base \
       env=RMPickPlaceCan-v0 tag=n2xT003 \
       method_loss=v0 init_temp=0.03

python train_dnn/run_algs.py alg=iql base=can_base \
       env=RMPickPlaceCan-v0 tag=n2xT005 \
       method_loss=v0 init_temp=0.05

python train_dnn/run_algs.py alg=iql base=can_base \
       env=RMPickPlaceCan-v0 tag=n2xT03 \
       method_loss=v0 init_temp=0.3

python train_dnn/run_algs.py alg=iql base=can_base \
       env=RMPickPlaceCan-v0 tag=n2xT03 \
       method_loss=v0 init_temp=0.1