python ../t5_experiment_2_eval.py --shots 8 \
                             --model_size 3B \
                             --support supp \
                             --input_len 512 \
                             --target_len 384 \
                             --lr 0.00002 \
                             --tpu_name "node-3"

./chain01XX.sh