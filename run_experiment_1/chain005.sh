python ../t5_experiment_1.py --shots all \
                             --model_size 3B \
                             --support all \
                             --input_len 512 \
                             --target_len 384 \
                             --lr 0.00002 \
                             --tpu_name "t5-tpu"

# ../run_baseline_0/chain004.sh