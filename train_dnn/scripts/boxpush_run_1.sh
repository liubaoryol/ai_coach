# IQL
python train_dnn/run_algs.py alg=iql base=boxpush_base \
       env=CleanupSingle-v0 tag=sv0 supervision=0.0 

python train_dnn/run_algs.py alg=iql base=boxpush_base \
       env=EnvMovers-v0 tag=sv0 supervision=0.0 

python train_dnn/run_algs.py alg=iql base=boxpush_base \
       env=EnvCleanup-v0 tag=sv0 supervision=0.0 
 
# OIQL
python train_dnn/run_algs.py alg=oiql base=MultiGoals2D_base \
       env=CleanupSingle-v0 tag=sv0 supervision=0.0 

python train_dnn/run_algs.py alg=oiql base=MultiGoals2D_base \
       env=EnvMovers-v0 tag=sv0 supervision=0.0 

python train_dnn/run_algs.py alg=oiql base=MultiGoals2D_base \
       env=EnvCleanup-v0 tag=sv0 supervision=0.0 
  