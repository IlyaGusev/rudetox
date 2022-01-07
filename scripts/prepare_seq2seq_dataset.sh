#!/bin/bash

DETOX_TRAIN_FILE="../data/russe_detox_2022/train.tsv";
DETOX_DEV_FILE="../data/russe_detox_2022/dev.tsv";

TRAIN_FILE="../data/seq2seq_train.jsonl";
VAL_FILE="../data/seq2seq_val.jsonl";

echo "Detox processing...";
python3.9 -m seq2seq.converters.detox \
    --input-file $DETOX_TRAIN_FILE \
    --output-file $TRAIN_FILE;

python3.9 -m seq2seq.converters.detox \
    --input-file $DETOX_DEV_FILE \
    --output-file $VAL_FILE;

echo "Train: $TRAIN_FILE";
echo "Val: $VAL_FILE";
