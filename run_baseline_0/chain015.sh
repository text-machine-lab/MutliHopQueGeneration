python ../t5_baseline_0.py --shots 256 \
                        --model_size 3B \
                        --support supp \
                        --input_len 512 \
                        --target_len 384 \
                        --lr 0.00002 \
                        --tpu_name "node-2"

./chain016.sh