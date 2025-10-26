#!/bin/bash
# run_experiments_small.sh —— 12 次左右
set -e
mkdir -p results
model="EEGNet"; optimizer="Adam"; batch=32; epochs=200; wd="1e-5"
declare -a cases=(
  "lr=0.001,dp=0.5,act=ReLU"
  "lr=0.001,dp=0.25,act=ReLU"
  "lr=0.0007,dp=0.5,act=ReLU"
  "lr=0.001,dp=0.5,act=LeakyReLU"
  "lr=0.0007,dp=0.25,act=LeakyReLU"
  "lr=0.001,dp=0.25,act=ELU,a=0.5"
  "lr=0.0007,dp=0.25,act=ELU,a=0.5"
  "lr=0.001,dp=0.5,act=ELU,a=0.5"
  "lr=0.001,dp=0.5,act=ELU,a=1.0"
  "lr=0.001,dp=0.5,act=ELU,a=1.5"
  "lr=0.0015,dp=0.5,act=ReLU"
  "lr=0.0003,dp=0.5,act=ReLU"
)
# run_deepconvnet_experiments.sh
# model="DeepConvNet"; optimizer="Adam"; batch=32; epochs=200; wd="1e-5"
# declare -a cases=(
#   "lr=0.0003,dp=0.5,act=ReLU"
#   "lr=0.0007,dp=0.5,act=ReLU"
#   "lr=0.0003,dp=0.75,act=ReLU"
#   "lr=0.0007,dp=0.5,act=LeakyReLU"
#   "lr=0.0003,dp=0.5,act=LeakyReLU"
#   "lr=0.0003,dp=0.5,act=ELU,a=1.0"
#   "lr=0.0003,dp=0.75,act=ELU,a=0.5"
# )

for c in "${cases[@]}"; do
  IFS=',' read -ra kv <<< "$c"; lr=; dp=; act=; a=
  for pair in "${kv[@]}"; do
    k="${pair%%=*}"; v="${pair##*=}"; case "$k" in
      lr) lr="$v";; dp) dp="$v";; act) act="$v";; a) a="$v";;
    esac
  done
  exp_id="${model}_opt${optimizer}_bs${batch}_lr${lr}_dp${dp}_ep${epochs}_wd${wd}_act${act}"
  extra=()
  if [[ "$act" == "ELU" ]]; then
    exp_id="${exp_id}_a${a}"
    extra+=(-elu_alpha "$a")
  fi
  echo ">>> $exp_id"
  uv run python main.py -model $model -optimizer $optimizer -batch_size $batch \
    -lr $lr -dropout $dp -num_epochs $epochs -weight_decay $wd \
    -activation $act "${extra[@]}" -experiment_id "$exp_id"
done
