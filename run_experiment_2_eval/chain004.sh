python ../t5_experiment_2_eval.py --shots 8 \
                             --model_size 3B \
                             --support all \
                             --input_len 512 \
                             --target_len 384 \
                             --lr 0.00002 \
                             --tpu_name "node-2"

./chain010.sh