# IIQL
python analysis/IIQL_results/infer_latent.py --alg "miql" \
      --env "MultiGoals2D_2-v0" --ndata 50 --logroot "result_lambda" \
      --modelpath "seed1Sv2/2023-10-05_08-32-45/model/iq_MultiGoals2D_2-v0_n50_l10_best"

python analysis/IIQL_results/infer_latent.py --alg "miql" \
      --env "MultiGoals2D_3-v0" --ndata 50 --logroot "result_lambda" \
      --modelpath "seed1Sv2/2023-10-05_16-42-51/model/iq_MultiGoals2D_3-v0_n50_l10_best"

python analysis/IIQL_results/infer_latent.py --alg "miql" \
      --env "MultiGoals2D_4-v0" --ndata 50 --logroot "result" \
      --modelpath "seed1Sv2/2023-10-05_11-54-45/model/iq_MultiGoals2D_4-v0_n50_l10_best"

python analysis/IIQL_results/infer_latent.py --alg "miql" \
      --env "MultiGoals2D_5-v0" --ndata 50 --logroot "result" \
      --modelpath "seed1Sv2/2023-10-05_14-14-48/model/iq_MultiGoals2D_5-v0_n50_l10_best"

python analysis/IIQL_results/infer_latent.py --alg "miql" \
      --env "CleanupSingle-v0" --ndata 50 --logroot "result_lambda" \
      --modelpath "sv2seed1/2023-10-05_22-09-19/model/iq_CleanupSingle-v0_n50_l10_best"

python analysis/IIQL_results/infer_latent.py --alg "miql" \
      --env "EnvMovers-v0" --ndata 22 --logroot "result_lambda" \
      --modelpath "sv2seed1/2023-10-05_15-57-38/model/iq_EnvMovers-v0_n44_l8_best"
#
# OGAIL
python analysis/IIQL_results/infer_latent.py --alg "ogail" \
      --env "MultiGoals2D_2-v0" --ndata 50 --logroot "result_desk" \
      --modelpath "seed1Sv2/2023-10-05_03-24-02/model/MultiGoals2D_2-v0_n50_l10_best.torch"

python analysis/IIQL_results/infer_latent.py --alg "ogail" \
      --env "MultiGoals2D_3-v0" --ndata 50 --logroot "result_desk" \
      --modelpath "seed1Sv2/2023-10-05_04-54-27/model/MultiGoals2D_3-v0_n50_l10_best.torch"

python analysis/IIQL_results/infer_latent.py --alg "ogail" \
      --env "MultiGoals2D_4-v0" --ndata 50 --logroot "result_desk" \
      --modelpath "seed1Sv2/2023-10-05_06-56-39/model/MultiGoals2D_4-v0_n50_l10_best.torch"

python analysis/IIQL_results/infer_latent.py --alg "ogail" \
      --env "MultiGoals2D_5-v0" --ndata 50 --logroot "result_desk" \
      --modelpath "seed1Sv2/2023-10-05_09-35-35/model/MultiGoals2D_5-v0_n50_l10_best.torch"

python analysis/IIQL_results/infer_latent.py --alg "ogail" \
      --env "CleanupSingle-v0" --ndata 50 --logroot "result_lambda" \
      --modelpath "sv2seed1/2023-10-06_15-50-25/model/CleanupSingle-v0_n50_l10_best.torch"

# python analysis/IIQL_results/infer_latent.py --alg "ogail" \
#       --env "EnvMovers-v0" --ndata 22 --logroot "result_lambda" \
#       --modelpath ""

# OIQL
python analysis/IIQL_results/infer_latent.py --alg "oiql" \
      --env "MultiGoals2D_2-v0" --ndata 50 --logroot "result_lambda" \
      --modelpath "seed1Sv2/2023-10-05_08-29-33/model/iq_MultiGoals2D_2-v0_n50_l10_best"

python analysis/IIQL_results/infer_latent.py --alg "oiql" \
      --env "MultiGoals2D_3-v0" --ndata 50 --logroot "result_desk" \
      --modelpath "seed1Sv2/2023-10-05_15-13-08/model/iq_MultiGoals2D_3-v0_n50_l10_best"

python analysis/IIQL_results/infer_latent.py --alg "oiql" \
      --env "MultiGoals2D_4-v0" --ndata 50 --logroot "result_lambda" \
      --modelpath "seed1Sv2/2023-10-05_17-07-29/model/iq_MultiGoals2D_4-v0_n50_l10_best"

python analysis/IIQL_results/infer_latent.py --alg "oiql" \
      --env "MultiGoals2D_5-v0" --ndata 50 --logroot "result_desk" \
      --modelpath "seed1Sv2/2023-10-05_14-24-06/model/iq_MultiGoals2D_5-v0_n50_l10_best"

python analysis/IIQL_results/infer_latent.py --alg "oiql" \
      --env "CleanupSingle-v0" --ndata 50 --logroot "result_lambda" \
      --modelpath "sv2seed1/2023-10-05_22-10-58/model/iq_CleanupSingle-v0_n50_l10_best"

python analysis/IIQL_results/infer_latent.py --alg "oiql" \
      --env "EnvMovers-v0" --ndata 22 --logroot "result_desk" \
      --modelpath "sv2seed1/2023-10-05_21-39-32/model/iq_EnvMovers-v0_n44_l8_best"



