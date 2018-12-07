#!/bin/bash
PROBLEM=languagemodel_lm1b32k
TMP_DIR=/datasets/BigLearning/aqiao/tmp
DATA_DIR=/datasets/BigLearning/aqiao/languagemodel_lm1b32k

MODEL=attention_lm_moe
HPARAMS=attention_lm_no_moe_small
TRAIN_STEPS=400

TRAIN_DIR=/proj/BigLearning/aqiao/t2t_train_single/$PROBLEM/$MODEL-$HPARAMS

mkdir -p $TRAIN_DIR

DBG_PROFILE=false

USE_NVPROF=false

if [ $USE_NVPROF == "true" ]
then
    /usr/local/cuda/bin/nvprof \
	--export-profile profile.nvvp \
	-f --print-summary \
	./tensor2tensor/bin/t2t-trainer \
	--data_dir=$DATA_DIR \
	--tmp_dir=$TMP_DIR \
	--problem=$PROBLEM \
	--model=$MODEL \
	--hparams_set=$HPARAMS \
	--output_dir=$TRAIN_DIR \
	--train_steps=$TRAIN_STEPS \
	--dbgprofile=$DBG_PROFILE
else
    ./tensor2tensor/bin/t2t-trainer \
	--data_dir=$DATA_DIR \
	--tmp_dir=$TMP_DIR \
	--problem=$PROBLEM \
	--model=$MODEL \
	--hparams_set=$HPARAMS \
	--output_dir=$TRAIN_DIR \
	--train_steps=$TRAIN_STEPS \
	--dbgprofile=$DBG_PROFILE
fi
