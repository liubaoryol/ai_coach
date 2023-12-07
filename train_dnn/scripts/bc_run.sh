# python train_dnn/run_algs.py env=Hopper-v2 alg=bc n_traj=1

# python train_dnn/run_algs.py env=HalfCheetah-v2 alg=bc n_traj=5

# python train_dnn/run_algs.py env=Walker2d-v2 alg=bc n_traj=5 

# python train_dnn/run_algs.py env=Ant-v2 alg=bc n_traj=5 

# python train_dnn/run_algs.py env=Humanoid-v2 alg=bc n_traj=5 

# python train_dnn/run_algs.py env=AntPush-v0-original alg=bc n_traj=253 

# python train_dnn/run_algs.py env=CleanupSingle-v0 alg=bc n_traj=50 

# python train_dnn/run_algs.py env=EnvMovers-v0 alg=bc n_traj=44

# python train_dnn/run_algs.py env=EnvCleanup-v0 alg=bc n_traj=66

python train_dnn/run_algs.py env=RMPickPlaceCan-v0 alg=bc n_traj=30 seed=0 \
       hidden_critic=[1,1] hidden_policy=[512,512,512] n_batches=10000


# python train_dnn/run_algs.py env=RMPickPlaceCan-v0 alg=bc n_traj=30 seed=2 \
#        hidden_critic=[512,512] hidden_policy=[512,512]
 

