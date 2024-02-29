
# MIQL
python train_dnn/run_algs.py alg=miql base=boxpush_base \
       env=EnvCleanup-v0 tag=sv2seed1 seed=1 supervision=0.2

python train_dnn/run_algs.py alg=miql base=boxpush_base \
       env=EnvMovers-v0 tag=sv2seed1 seed=1 supervision=0.2

python train_dnn/run_algs.py alg=miql base=boxpush_base \
       env=CleanupSingle-v0 tag=sv2seed1 seed=1 supervision=0.2
  
 
# OIQL
python train_dnn/run_algs.py alg=oiql base=MultiGoals2D_base \
       env=CleanupSingle-v0 tag=sv2seed1 seed=1 supervision=0.2

python train_dnn/run_algs.py alg=oiql base=MultiGoals2D_base \
       env=EnvMovers-v0 tag=sv2seed1 seed=1 supervision=0.2

python train_dnn/run_algs.py alg=oiql base=MultiGoals2D_base \
       env=EnvCleanup-v0 tag=sv2seed1 seed=1 supervision=0.2 

