# IQL


python train_dnn/run_algs.py alg=iql base=can_base offline=True \
       env=RMPickPlaceCan-v0 tag=512x3T001 eval_interval=10000 \
       mini_batch_size=256 max_explore_step=1e5 \
       hidden_critic=[512,512,512] hidden_policy=[512,512,512] \
       optimizer_lr_critic=3.e-4 optimizer_lr_policy=3.e-5 \
       method_div=chi method_regularize=False \
       init_temp=0.01 iql_single_critic=False
