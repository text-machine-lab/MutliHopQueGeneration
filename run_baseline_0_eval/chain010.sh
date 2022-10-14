python ../t5_baseline_0_eval.py --shots 8 \
                        --model_size 3B \
                        --support supp \
                        --input_len 512 \
                        --target_len 384 \
                        --lr 0.00002 \
                        --tpu_name "node-3" \
                        --eval_checkpoint 1001500

./chain011.sh