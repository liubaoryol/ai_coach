# MIQL
python train_dnn/run_algs.py alg=miql base=MultiGoals2D_base \
       env=CleanupSingle-v0 tag=sv0 supervision=0.0 

python train_dnn/run_algs.py alg=miql base=MultiGoals2D_base \
       env=EnvMovers-v0 tag=sv0 supervision=0.0 

python train_dnn/run_algs.py alg=miql base=MultiGoals2D_base \
       env=EnvCleanup-v0 tag=sv0 supervision=0.0 
  
# OGAIL
python train_dnn/run_algs.py alg=ogail base=MultiGoals2D_base \
       env=CleanupSingle-v0 tag=sv0 supervision=0.0 

python train_dnn/run_algs.py alg=ogail base=MultiGoals2D_base \
       env=EnvMovers-v0 tag=sv0 supervision=0.0 

python train_dnn/run_algs.py alg=ogail base=MultiGoals2D_base \
       env=EnvCleanup-v0 tag=sv0 supervision=0.0 
    