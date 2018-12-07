#!/bin/bash
PROBLEM=languagemodel_lm1b32k
TMP_DIR=/datasets/BigLearning/aqiao/tmp
DATA_DIR=/datasets/BigLearning/aqiao/languagemodel_lm1b32k

#PROBLEM=image_mnist
#DATA_DIR=/datasets/BigLearning/aqiao/image_mnist

mkdir -p $TMP_DIR $DATA_DIR

./tensor2tensor/bin/t2t-datagen \
    --data_dir=$DATA_DIR \
    --tmp_dir=$TMP_DIR \
    --problem=$PROBLEM
