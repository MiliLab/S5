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

##############Default values##################
GPUS_DEFAULT=2
BACKBONE_DEFAULT="vit_b_moe"
LOAD_NETWORK_DEFAULT=True
RESUME_DEFAULT="Your/Path/vit_b_s4p_upernet.pth"
##############################################

GPUS=${1:-$GPUS_DEFAULT}
PORT=${2:-29501}
BACKBONE=${3:-$BACKBONE_DEFAULT}
LOAD_NETWORK=${4:-$LOAD_NETWORK_DEFAULT}
RESUME=${5:-$RESUME_DEFAULT}


export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $(($GPUS-1)))

python -m torch.distributed.launch \
    --nproc_per_node=${GPUS} \
    --master_addr=localhost \
    --master_port=${PORT} \
    --use_env ${method}.py \
    --config=${config} \
    --backbone ${BACKBONE} \
    --init_backbone "none" \
    --load_network ${LOAD_NETWORK} \
    --experts 4 \
    --part 256 \
    --resume ${RESUME} \
    --vaihingen-id-path ${vaihingen_id_path} \
    --potsdam-id-path ${potsdam_id_path} \
    --openearthmap-id-path ${openearthmap_id_path} \
    --loveda-id-path ${loveda_id_path} \
    --save-path ${save_path} \
    2>&1 | tee ${save_path}/${now}.log
