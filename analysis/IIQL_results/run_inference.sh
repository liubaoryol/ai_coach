
python analysis/IIQL_results/infer_latent.py --alg "miql" \
      --env "MultiGoals2D_2-v0" --ndata 50 \
      --modelpath "Ttx001Tpi001tol5Sv2/2023-09-20_10-23-11/model/iq_MultiGoals2D_2-v0_n50_l10_best"

python analysis/IIQL_results/infer_latent.py --alg "miql" \
      --env "MultiGoals2D_3-v0" --ndata 50 \
      --modelpath "Ttx001Tpi001tol5Sv2/2023-09-20_16-43-42/model/iq_MultiGoals2D_3-v0_n50_l10_best"

python analysis/IIQL_results/infer_latent.py --alg "miql" \
      --env "MultiGoals2D_4-v0" --ndata 50 \
      --modelpath "Ttx001Tpi001tol5Sv2/2023-09-20_23-08-50/model/iq_MultiGoals2D_4-v0_n50_l10_best"

python analysis/IIQL_results/infer_latent.py --alg "miql" \
      --env "MultiGoals2D_5-v0" --ndata 50 \
      --modelpath "Ttx001Tpi001tol5Sv2/2023-09-21_06-33-56/model/iq_MultiGoals2D_5-v0_n50_l10_best"

python analysis/IIQL_results/infer_latent.py --alg "ogail" \
      --env "MultiGoals2D_2-v0" --ndata 50 \
      --modelpath "tol5Sv2/2023-09-20_15-35-27/model/MultiGoals2D_2-v0_n50_l10_best.torch"

python analysis/IIQL_results/infer_latent.py --alg "ogail" \
      --env "MultiGoals2D_3-v0" --ndata 50 \
      --modelpath "tol5Sv2/2023-09-21_00-20-45/model/MultiGoals2D_3-v0_n50_l10_best.torch"

python analysis/IIQL_results/infer_latent.py --alg "ogail" \
      --env "MultiGoals2D_4-v0" --ndata 50 \
      --modelpath "tol5Sv2/2023-09-21_09-12-36/model/MultiGoals2D_4-v0_n50_l10_best.torch"

python analysis/IIQL_results/infer_latent.py --alg "ogail" \
      --env "MultiGoals2D_5-v0" --ndata 50 \
      --modelpath "tol5Sv2/2023-09-21_19-11-56/model/MultiGoals2D_5-v0_n50_l10_best.torch"


