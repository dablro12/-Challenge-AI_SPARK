nohup python ./run.py --model "unet" \
                      --version "v1" \
                      --cuda "0" \
                      --ts_batch_size 80 \
                      --vs_batch_size 40 \
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