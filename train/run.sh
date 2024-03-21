nohup python ./run.py --model "R2Att-Unet" \
                      --version "v4" \
                      --cuda "0" \
                      --ts_batch_size 5\
                      --vs_batch_size 1\
                      --epochs 200 \
                      --loss "bce" \
                      --optimizer "AdamW" \
                      --learning_rate 0.001 \
                      --scheduler "lambda" \
                      --pretrain "no" \
                      --pretrained_model "practice" \
                      --error_signal no \
                      --wandb "yes" \
                      > output.log 2>&1 &