python ../t5_experiment_1.py --shots 128 \
                             --model_size 3B \
                             --support supp \
                             --input_len 512 \
                             --target_len 384 \
                             --lr 0.00002 \
                             --tpu_name "node-1"

./chain011.sh