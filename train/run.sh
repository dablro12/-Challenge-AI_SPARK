nohup python ./run.py --model "unet" \
                      --version "v8" \
                      --cuda "0" \
                      --ts_batch_size 40\
                      --vs_batch_size 4\
                      --epochs 200 \
                      --loss "bce" \
                      --optimizer "AdamW" \
                      --learning_rate 0.002 \
                      --scheduler "lambda" \
                      --pretrain "no" \
                      --pretrained_model "practice" \
                      --error_signal no \
                      --wandb "yes" \
                      > output.log 2>&1 &