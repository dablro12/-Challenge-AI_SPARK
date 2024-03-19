nohup python ./run.py --model "Att-Unet" \
                      --version "v3" \
                      --cuda "0" \
                      --ts_batch_size 12\
                      --vs_batch_size 2 \
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