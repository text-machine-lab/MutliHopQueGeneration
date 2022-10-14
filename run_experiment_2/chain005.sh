python ../t5_experiment_2.py --shots 16 \
                             --model_size 3B \
                             --support all \
                             --input_len 512 \
                             --target_len 384 \
                             --lr 0.00002 \
                             --tpu_name "node-1"

./chain006.sh