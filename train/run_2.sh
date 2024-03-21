nohup python ./run_2.py --model "unet++" \
                      --version "v5" \
                      --cuda "0" \
                      --ts_batch_size 4 \
                      --vs_batch_size 1 \
                      --epochs 200 \
                      --loss "bce" \
                      --optimizer "AdamW" \
                      --learning_rate 0.001 \
                      --scheduler "lambda" \
                      --pretrain "no" \
                      --pretrained_model "practice" \
                      --error_signal no \
                      --wandb "yes
                      " \
                      > output_2.log 2>&1 &