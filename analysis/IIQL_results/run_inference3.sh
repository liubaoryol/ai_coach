# IIQL
python analysis/IIQL_results/infer_latent.py --alg "miql" \
      --env "MultiGoals2D_2-v0" --ndata 50 --logroot "result_lambda" \
      --modelpath "seed2Sv2/2023-10-05_08-33-45/model/iq_MultiGoals2D_2-v0_n50_l10_best"

python analysis/IIQL_results/infer_latent.py --alg "miql" \
      --env "MultiGoals2D_3-v0" --ndata 50 --logroot "result_lambda" \
      --modelpath "seed2Sv2/2023-10-05_16-44-37/model/iq_MultiGoals2D_3-v0_n50_l10_best"

python analysis/IIQL_results/infer_latent.py --alg "miql" \
      --env "MultiGoals2D_4-v0" --ndata 50 --logroot "result_desk" \
      --modelpath "seed2Sv2/2023-10-05_11-58-40/model/iq_MultiGoals2D_4-v0_n50_l10_best"

python analysis/IIQL_results/infer_latent.py --alg "miql" \
      --env "MultiGoals2D_5-v0" --ndata 50 --logroot "result_desk" \
      --modelpath "seed2Sv2/2023-10-05_14-27-17/model/iq_MultiGoals2D_5-v0_n50_l10_best"

python analysis/IIQL_results/infer_latent.py --alg "miql" \
      --env "CleanupSingle-v0" --ndata 50 --logroot "result_lambda" \
      --modelpath "sv2seed2/2023-10-05_22-10-05/model/iq_CleanupSingle-v0_n50_l10_best"

python analysis/IIQL_results/infer_latent.py --alg "miql" \
      --env "EnvMovers-v0" --ndata 22 --logroot "result_desk" \
      --modelpath "sv2seed2/2023-10-05_21-55-01/model/iq_EnvMovers-v0_n44_l8_best"

# OGAIL
python analysis/IIQL_results/infer_latent.py --alg "ogail" \
      --env "MultiGoals2D_2-v0" --ndata 50 --logroot "result_desk" \
      --modelpath "seed2Sv2/2023-10-05_03-25-24/model/MultiGoals2D_2-v0_n50_l10_best.torch"

python analysis/IIQL_results/infer_latent.py --alg "ogail" \
      --env "MultiGoals2D_3-v0" --ndata 50 --logroot "result_desk" \
      --modelpath "seed2Sv2/2023-10-05_04-55-55/model/MultiGoals2D_3-v0_n50_l10_best.torch"

python analysis/IIQL_results/infer_latent.py --alg "ogail" \
      --env "MultiGoals2D_4-v0" --ndata 50 --logroot "result_desk" \
      --modelpath "seed2Sv2/2023-10-05_07-00-12/model/MultiGoals2D_4-v0_n50_l10_best.torch"

python analysis/IIQL_results/infer_latent.py --alg "ogail" \
      --env "MultiGoals2D_5-v0" --ndata 50 --logroot "result_desk" \
      --modelpath "seed2Sv2/2023-10-05_10-36-17/model/MultiGoals2D_5-v0_n50_l10_best.torch"

python analysis/IIQL_results/infer_latent.py --alg "ogail" \
      --env "CleanupSingle-v0" --ndata 50 --logroot "result_lambda" \
      --modelpath "sv2seed2/2023-10-06_15-50-44/model/CleanupSingle-v0_n50_l10_best.torch"

# python analysis/IIQL_results/infer_latent.py --alg "ogail" \
#       --env "EnvMovers-v0" --ndata 22 --logroot "result_lambda" \
#       --modelpath ""

# OIQL
python analysis/IIQL_results/infer_latent.py --alg "oiql" \
      --env "MultiGoals2D_2-v0" --ndata 50 --logroot "result_lambda" \
      --modelpath "seed2Sv2/2023-10-05_08-30-31/model/iq_MultiGoals2D_2-v0_n50_l10_best"

python analysis/IIQL_results/infer_latent.py --alg "oiql" \
      --env "MultiGoals2D_3-v0" --ndata 50 --logroot "result_lambda" \
      --modelpath "seed2Sv2/2023-10-06_01-16-06/model/iq_MultiGoals2D_3-v0_n50_l10_best"

python analysis/IIQL_results/infer_latent.py --alg "oiql" \
      --env "MultiGoals2D_4-v0" --ndata 50 --logroot "result_lambda" \
      --modelpath "seed2Sv2/2023-10-05_17-08-17/model/iq_MultiGoals2D_4-v0_n50_l10_best"

python analysis/IIQL_results/infer_latent.py --alg "oiql" \
      --env "MultiGoals2D_5-v0" --ndata 50 --logroot "result_desk" \
      --modelpath "seed2Sv2/2023-10-05_14-25-30/model/iq_MultiGoals2D_5-v0_n50_l10_best"

python analysis/IIQL_results/infer_latent.py --alg "oiql" \
      --env "CleanupSingle-v0" --ndata 50 --logroot "result_lambda" \
      --modelpath "sv2seed2/2023-10-05_22-11-30/model/iq_CleanupSingle-v0_n50_l10_best"

python analysis/IIQL_results/infer_latent.py --alg "oiql" \
      --env "EnvMovers-v0" --ndata 22 --logroot "result_desk" \
      --modelpath "sv2seed2/2023-10-05_21-41-11/model/iq_EnvMovers-v0_n44_l8_best"



