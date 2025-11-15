#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

dataset='pretrain'
method='s4_pretrain'
backbone=$3

config=configs/${dataset}.yaml
labeled_id_path=splits/$dataset/labeled/labeled.txt
unlabeled_id_path=splits/$dataset/unlabeled/RS4P-1M.txt
save_path=exp/$method/$backbone/

mkdir -p $save_path
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_addr=localhost \
    --master_port=$2 \
    --use_env $method.py \
    --backbone $backbone \
    --init_backbone $4 \
    --decoder 'upernet' \
    --config=$config \
    --labeled-id-path $labeled_id_path \
    --unlabeled-id-path $unlabeled_id_path \
    --save-path $save_path \
    --port $2 2>&1 | tee $save_path/$now.log
