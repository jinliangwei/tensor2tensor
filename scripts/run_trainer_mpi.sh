#!/bin/bash

PROBLEM=languagemodel_lm1b8k_packed
TMP_DIR=/datasets/BigLearning/aqiao/tmp
DATA_DIR=/datasets/BigLearning/aqiao/languagemodel_lm1b8k_packed

MODEL=mtf_transformer
HPARAMS=mtf_transformer_tiny_lm_moe
TRAIN_STEPS=1000

MESH_SIZE=$OMPI_COMM_WORLD_SIZE
#MESH_SHAPE="batch:1;model:$MESH_SIZE"
#LAYOUT="batch:batch;vocab:model;d_ff:model;heads:model;experts:model"
MESH_SHAPE="all:$MESH_SIZE"
LAYOUT="batch:all;experts:all"
BATCH_SIZE=200
NUM_EXPERTS=80

TRAIN_DIR=/proj/BigLearning/aqiao/t2t_train_single/$PROBLEM/$MODEL-$HPARAMS

mkdir -p $TRAIN_DIR

unset http_proxy
unset https_proxy
. $HOME/miniconda3/etc/profile.d/conda.sh
conda activate moe
export PYTHONPATH=$PWD:$PYTHONPATH

PS_PORT=5555
MASTER_PORT=5000
PS_LIST=$(eval echo \\\"h{0..$((MESH_SIZE-1))}:$PS_PORT\\\" | tr ' ' ,)

TF_CONFIG='{"cluster": {"ps": ['$PS_LIST'], "master": ["h0:'$MASTER_PORT'"]}'
TF_CONFIG=$TF_CONFIG', "environment": "cloud"'

if [ $OMPI_COMM_WORLD_RANK -eq 0 ]; then
  CUDA_VISIBLE_DEVICES='' \
  TF_CONFIG=$TF_CONFIG', "task": {"type": "master", "index": 0}}' \
  ./tensor2tensor/bin/t2t-trainer \
    --master=grpc://h0:$MASTER_PORT \
    --ps_replicas=$MESH_SIZE \
    --worker_replicas=1 \
    --worker_gpu=0 \
    --worker_id=0 \
    --ps_gpu=1 \
    --sync \
    --schedule=train \
    --worker_job='/job:master' \
    --data_dir=$DATA_DIR \
    --tmp_dir=$TMP_DIR \
    --problem=$PROBLEM \
    --model=$MODEL \
    --hparams_set=$HPARAMS \
    --hparams="mesh_shape=$MESH_SHAPE,layout=$LAYOUT,moe_num_experts=$NUM_EXPERTS,batch_size=$BATCH_SIZE" \
    --output_dir=$TRAIN_DIR \
    --train_steps=$TRAIN_STEPS &
fi

if [ $OMPI_COMM_WORLD_RANK -lt $MESH_SIZE ]; then
  export TF_MEM_LOGGER_PATH_PREFIX=/users/aqiao/moe/tensor2tensor/memlog
  export LD_LIBRARY_PATH=/users/aqiao/cuda-9.0/lib64:$LD_LIBRARY_PATH
  TF_CONFIG=$TF_CONFIG', "task": {"type": "ps", "index": '$OMPI_COMM_WORLD_RANK'}}' \
  ./tensor2tensor/bin/t2t-trainer \
    --schedule=run_std_server \
    --data_dir=$DATA_DIR \
    --tmp_dir=$TMP_DIR \
    --problem=$PROBLEM \
    --model=$MODEL \
    --hparams_set=$HPARAMS \
    --hparams="mesh_shape=$MESH_SHAPE,layout=$LAYOUT,moe_num_experts=$NUM_EXPERTS,batch_size=$BATCH_SIZE" \
    --output_dir=$TRAIN_DIR \
    --train_steps=$TRAIN_STEPS
fi
