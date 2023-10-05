# IIQL
# python analysis/IIQL_results/infer_latent.py --alg "miql" \
#       --env "MultiGoals2D_2-v0" --ndata 50 --logroot "result" \
#       --modelpath "Ttx001Tpi001tol5Sv2/2023-09-20_10-23-11/model/iq_MultiGoals2D_2-v0_n50_l10_best"

python analysis/IIQL_results/infer_latent.py --alg "miql" \
      --env "MultiGoals2D_3-v0" --ndata 50 --logroot "result" \
      --modelpath "Ttx001Tpi001tol5Sv2/2023-09-20_16-43-42/model/iq_MultiGoals2D_3-v0_n50_l10_best"

# python analysis/IIQL_results/infer_latent.py --alg "miql" \
#       --env "MultiGoals2D_4-v0" --ndata 50 --logroot "result" \
#       --modelpath "Ttx001Tpi001tol5Sv2/2023-09-20_23-08-50/model/iq_MultiGoals2D_4-v0_n50_l10_best"

python analysis/IIQL_results/infer_latent.py --alg "miql" \
      --env "MultiGoals2D_5-v0" --ndata 50 --logroot "result" \
      --modelpath "Ttx001Tpi001tol5Sv2/2023-09-21_06-33-56/model/iq_MultiGoals2D_5-v0_n50_l10_best"

python analysis/IIQL_results/infer_latent.py --alg "miql" \
      --env "CleanupSingle-v0" --ndata 50 --logroot "result_lambda" \
      --modelpath "sv2/2023-09-24_07-22-36/model/iq_CleanupSingle-v0_n50_l10_best"

python analysis/IIQL_results/infer_latent.py --alg "miql" \
      --env "EnvMovers-v0" --ndata 22 --logroot "result_lambda" \
      --modelpath "sv2/2023-09-24_05-11-53/model/iq_EnvMovers-v0_n44_l8_best"
#
# OGAIL
# python analysis/IIQL_results/infer_latent.py --alg "ogail" \
#       --env "MultiGoals2D_2-v0" --ndata 50 --logroot "result" \
#       --modelpath "tol5Sv2/2023-09-20_15-35-27/model/MultiGoals2D_2-v0_n50_l10_best.torch"

python analysis/IIQL_results/infer_latent.py --alg "ogail" \
      --env "MultiGoals2D_3-v0" --ndata 50 --logroot "result" \
      --modelpath "tol5Sv2/2023-09-21_00-20-45/model/MultiGoals2D_3-v0_n50_l10_best.torch"

# python analysis/IIQL_results/infer_latent.py --alg "ogail" \
#       --env "MultiGoals2D_4-v0" --ndata 50 --logroot "result" \
#       --modelpath "tol5Sv2/2023-09-21_09-12-36/model/MultiGoals2D_4-v0_n50_l10_best.torch"

python analysis/IIQL_results/infer_latent.py --alg "ogail" \
      --env "MultiGoals2D_5-v0" --ndata 50 --logroot "result" \
      --modelpath "tol5Sv2/2023-09-21_19-11-56/model/MultiGoals2D_5-v0_n50_l10_best.torch"

# OIQL
# python analysis/IIQL_results/infer_latent.py --alg "oiql" \
#       --env "MultiGoals2D_2-v0" --ndata 50 --logroot "result_lambda" \
#       --modelpath "T001tol5Sv2/2023-09-21_15-43-48/model/iq_MultiGoals2D_2-v0_n50_l10_best"

python analysis/IIQL_results/infer_latent.py --alg "oiql" \
      --env "MultiGoals2D_3-v0" --ndata 50 --logroot "result_lambda" \
      --modelpath "T001tol5Sv2/2023-09-21_15-45-55/model/iq_MultiGoals2D_3-v0_n50_l10_best"

# python analysis/IIQL_results/infer_latent.py --alg "oiql" \
#       --env "MultiGoals2D_4-v0" --ndata 50 --logroot "result_lambda" \
#       --modelpath "T001tol5Sv2/2023-09-21_15-46-20/model/iq_MultiGoals2D_4-v0_n50_l10_best"

python analysis/IIQL_results/infer_latent.py --alg "oiql" \
      --env "MultiGoals2D_5-v0" --ndata 50 --logroot "result_lambda" \
      --modelpath "T001tol5Sv2/2023-09-21_15-46-40/model/iq_MultiGoals2D_5-v0_n50_l10_best"

python analysis/IIQL_results/infer_latent.py --alg "oiql" \
      --env "CleanupSingle-v0" --ndata 50 --logroot "result_lambda" \
      --modelpath "sv2/2023-09-24_08-03-29/model/iq_CleanupSingle-v0_n50_l10_best"

python analysis/IIQL_results/infer_latent.py --alg "oiql" \
      --env "EnvMovers-v0" --ndata 22 --logroot "result_lambda" \
      --modelpath "sv2/2023-09-24_06-05-01/model/iq_EnvMovers-v0_n44_l8_best"



