

# python analysis/IIQL_results/infer_latent.py --alg "ogail" \
#       --env "CleanupSingle-v0" --ndata 50 --logroot "result_lambda" \
#       --modelpath "sv2seed0/2023-10-06_15-49-29/model/CleanupSingle-v0_n50_l10_best.torch"

# python analysis/IIQL_results/infer_latent.py --alg "ogail" \
#       --env "CleanupSingle-v0" --ndata 50 --logroot "result_lambda" \
#       --modelpath "sv2seed1/2023-10-06_15-50-25/model/CleanupSingle-v0_n50_l10_best.torch"

# python analysis/IIQL_results/infer_latent.py --alg "ogail" \
#       --env "CleanupSingle-v0" --ndata 50 --logroot "result_lambda" \
#       --modelpath "sv2seed2/2023-10-06_15-50-44/model/CleanupSingle-v0_n50_l10_best.torch"

python analysis/IIQL_results/infer_latent.py --alg "ogail" \
      --env "EnvMovers-v0" --ndata 22 --logroot "result" \
      --modelpath "sv2seed0/2023-10-06_15-51-37/model/EnvMovers-v0_n44_l8_best.torch"

python analysis/IIQL_results/infer_latent.py --alg "ogail" \
      --env "EnvMovers-v0" --ndata 22 --logroot "result" \
      --modelpath "sv2seed2/2023-10-06_15-53-08/model/EnvMovers-v0_n44_l8_best.torch"
