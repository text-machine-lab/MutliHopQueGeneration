python ../t5_baseline_0_eval.py --shots 16 \
                        --model_size 3B \
                        --support all \
                        --input_len 512 \
                        --target_len 384 \
                        --lr 0.00002 \
                        --tpu_name "node-3" \
                        --eval_checkpoint 1002000

./chain002.sh