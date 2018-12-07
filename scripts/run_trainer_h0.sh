#!/bin/bash
PROBLEM=languagemodel_lm1b8k_packed
TMP_DIR=/datasets/BigLearning/aqiao/tmp
DATA_DIR=/datasets/BigLearning/aqiao/languagemodel_lm1b8k_packed

MODEL=mtf_transformer
HPARAMS=mtf_transformer_tiny_lm
TRAIN_STEPS=4000
MESH_SHAPE="batch:1;model:2"

TRAIN_DIR=/proj/BigLearning/aqiao/t2t_train_single/$PROBLEM/$MODEL-$HPARAMS

mkdir -p $TRAIN_DIR

DBG_PROFILE=false

USE_NVPROF=false

export TF_CONFIG='{"cluster": {"ps": ["h1:5555", "h2:5555"], "master": ["h0:5555"]}, "task": {"type": "master", "index": 0}, "environment": "cloud"}'	

if [ $USE_NVPROF == "true" ]
then
    /usr/local/cuda/bin/nvprof \
	--export-profile profile.nvvp \
	-f --print-summary \
	./tensor2tensor/bin/t2t-trainer --master=grpc://h0:5555 --ps_replicas=2 --worker_replicas=1 --worker_gpu=0 --worker_id=0 --ps_gpu=1 --sync --schedule=train --worker_job='/job:master' \
	--data_dir=$DATA_DIR \
	--tmp_dir=$TMP_DIR \
	--problem=$PROBLEM \
	--model=$MODEL \
	--hparams_set=$HPARAMS \
  --hparams="mesh_shape=$MESH_SHAPE" \
	--output_dir=$TRAIN_DIR \
	--train_steps=$TRAIN_STEPS \
	--dbgprofile=$DBG_PROFILE
else
    ./tensor2tensor/bin/t2t-trainer --master=grpc://h0:5555 --ps_replicas=2 --worker_replicas=1 --worker_gpu=0 --worker_id=0 --ps_gpu=1 --sync --schedule=train --worker_job='/job:master' \
	--data_dir=$DATA_DIR \
	--tmp_dir=$TMP_DIR \
	--problem=$PROBLEM \
	--model=$MODEL \
	--hparams_set=$HPARAMS \
  --hparams="mesh_shape=$MESH_SHAPE" \
	--output_dir=$TRAIN_DIR \
	--train_steps=$TRAIN_STEPS \
	--dbgprofile=$DBG_PROFILE
fi
