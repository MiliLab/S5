#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

dataset="rsseg"
method="md_finetune"
exp="vit_b_moe"
split="all"
config="configs/${dataset}.yaml"

vaihingen_id_path="splits/vaihingen/${split}/labeled.txt"
potsdam_id_path="splits/potsdam/${split}/labeled.txt"
openearthmap_id_path="splits/openearthmap/${split}/labeled.txt"
loveda_id_path="splits/loveda/${split}/labeled.txt"

save_path="exp/${dataset}/${method}/${exp}/${split}/"
mkdir -p "$save_path"

GPUS=${GPUS:-2}
BACKBONE=${BACKBONE:-"vit_b_moe"}
INIT_BACKBONE=${INIT_BACKBONE:-"none"}
EXPERTS=${EXPERTS:-4}
PART=${PART:-256}
RESUME=${RESUME:-"data_root: 'Your/Path/vit_b_s4p_upernet"}
LOAD_NETWORK=${LOAD_NETWORK:-True}
PORT=${PORT:-29501}

export CUDA_VISIBLE_DEVICES=0,1

python -m torch.distributed.launch \
    --nproc_per_node=${GPUS} \
    --master_addr=localhost \
    --master_port=${PORT} \
    --use_env ${method}.py \
    --config=${config} \
    --backbone ${BACKBONE} \
    --init_backbone ${INIT_BACKBONE} \
    --load_network ${LOAD_NETWORK} \
    --experts ${EXPERTS} \
    --part ${PART} \
    --resume ${RESUME} \
    --vaihingen-id-path ${vaihingen_id_path} \
    --potsdam-id-path ${potsdam_id_path} \
    --openearthmap-id-path ${openearthmap_id_path} \
    --loveda-id-path ${loveda_id_path} \
    --save-path ${save_path} \
    --port ${PORT} 2>&1 | tee ${save_path}/${now}.log
