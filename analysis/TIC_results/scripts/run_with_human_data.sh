python analysis/TIC_results/two_agent_domain_save_models.py \
      --domain=movers --synthetic=False --num-training-data=160 \
      --supervsion=0.3 --use-true-tx=False --gen-trainset=False \
      --beta-pi=0.01 --beta-tx=0.01 --tx-dependency=FTTT

python analysis/TIC_results/two_agent_domain_save_models.py \
      --domain=rescue_2 --synthetic=False --num-training-data=160 \
      --supervsion=0.3 --use-true-tx=False --gen-trainset=False \
      --beta-pi=0.01 --beta-tx=0.01 --tx-dependency=FTTT

python analysis/TIC_results/save_merged_v_values.py \
      --domain=movers --iteration=150 --num-train=160 --supervsion=0.3 \
      --humandata=True

python analysis/TIC_results/save_merged_v_values.py \
      --domain=rescue_2 --iteration=30 --num-train=160 --supervsion=0.3 \
      --humandata=True
