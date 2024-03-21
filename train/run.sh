nohup python ./run.py --model "manet" \
                      --version "v6" \
                      --cuda "0" \
                      --ts_batch_size 64\
                      --vs_batch_size 8\
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