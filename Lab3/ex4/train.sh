#!/bin/bash
DATASET=("../no_exp/cropped")
# Training Script
MODEL=("efficientnet_b2")
LOSS_TYPE=("wce" "focal" "label_smooth")
OPTIMIZER="adamw"
SCHEDULER="cosine"
LR=3e-4
WD=1e-4
BATCH_SIZE=32
RESIZE=384
NUM_EPOCHS=50
PRED_WEIGHTS="1.0,1.0,1.0,1.0"
USE_AMP="--use_amp"

train_csv="../csv/train_data_with_pseudo_bacteria_virus.csv"

# Loop through each model and train
for model in "${MODEL[@]}"; do
  for dataset in "${DATASET[@]}"; do
    for loss in "${LOSS_TYPE[@]}"; do
      python train.py \
        --dataset $dataset \
        --model $model \
        --loss_type $loss \
        --optimizer $OPTIMIZER \
        --scheduler $SCHEDULER \
        --lr $LR \
        --wd $WD \
        --batch_size $BATCH_SIZE \
        --resize $RESIZE \
        --num_epochs $NUM_EPOCHS \
        --pred_weights $PRED_WEIGHTS \
        $USE_AMP \
        --train_csv $train_csv \
        --experiment_id "${model}_$(basename $dataset)_${loss}_pseudo_virus_bacteria"
    done
  done
done