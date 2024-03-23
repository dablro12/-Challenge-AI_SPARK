nohup python ./run.py --model "swinunet" \
                      --version "v12" \
                      --cuda "0" \
                      --ts_batch_size 128\
                      --vs_batch_size 16\
                      --epochs 200 \
                      --loss "bcewithlogits" \
                      --optimizer "AdamW" \
                      --learning_rate 0.0002 \
                      --scheduler "lambda" \
                      --pretrain "no" \
                      --pretrained_model "practice" \
                      --error_signal no \
                      --wandb "yes" \
                      > output.log 2>&1 &