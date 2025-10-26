#!/bin/bash
# run_experiments.sh
# Usage: uv run bash run_experiments.sh
# Optimized hyperparameter sweep for EEGNet

mkdir -p results

model="EEGNet"
optimizers=("Adam" "AdamW")
batch_sizes=(16 32 64)
lrs=(0.0003 0.0007 0.001 0.0015)
dropouts=(0.25 0.5)
epochs_list=(100 150 200)
weight_decay=("1e-4" "1e-5")

for opt in "${optimizers[@]}"; do
  for bs in "${batch_sizes[@]}"; do
    for lr in "${lrs[@]}"; do
      for dp in "${dropouts[@]}"; do
        for ep in "${epochs_list[@]}"; do
          for wd in "${weight_decay[@]}"; do
            exp_id="${model}_opt${opt}_bs${bs}_lr${lr}_dp${dp}_ep${ep}_wd${wd}"
            echo "==============================================="
            echo " Running Experiment: $exp_id "
            echo "==============================================="
            uv run python main.py \
            -model $model \
            -optimizer $opt \
            -batch_size $bs \
            -lr $lr \
            -dropout $dp \
            -num_epochs $ep \
            -weight_decay $wd \
            -experiment_id $exp_id

          done
        done
      done
    done
  done
done
