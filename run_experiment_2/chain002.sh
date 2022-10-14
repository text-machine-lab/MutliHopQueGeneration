python ../t5_experiment_2.py --shots 128 \
                             --model_size 11B \
                             --support supp \
                             --input_len 512 \
                             --target_len 384 \
                             --lr 0.00002 \
                             --tpu_name "node-2"

../run_experiment_2_eval/chain002.sh