python ../t5_experiment_2.py --shots 8 \
                             --model_size 3B \
                             --support supp \
                             --input_len 512 \
                             --target_len 384 \
                             --lr 0.00002 \
                             --tpu_name "node-2"

./chain01X.sh