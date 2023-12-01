#!/bin/bash

# Usage : ./train_loop.sh $model_name $n_epochs

model_name=${1}
n_epochs=${2-50}
lr=0.001
wd=0.0001

random_seed=(0 10 42 100 256 512 800 1024 2500 5000)

for seed in ${random_seed[@]}; do {
python train_cnn.py --model_name "${model_name}_${seed}"  --learning_rate $lr --weight_decay $wd --scheduler False --n_epochs 50 --batch_size 512 --dropout_conv 0.0 --dropout_fc 0.0 --train_pct 100 --holdout_pct 0 --seed $seed
} done
