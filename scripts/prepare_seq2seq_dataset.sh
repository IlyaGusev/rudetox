#!/bin/bash
set -e

DETOX_TRAIN_FILE="data/detox_train.tsv";
DETOX_DEV_FILE="data/detox_dev.tsv";

TRAIN_FILE="data/seq2seq_train.jsonl";
VAL_FILE="data/seq2seq_val.jsonl";

cd rudetox;

echo "Detox processing...";
python3 -m seq2seq.converters.detox \
    --input-file ../$DETOX_TRAIN_FILE \
    --output-file ../$TRAIN_FILE;

python3 -m seq2seq.converters.detox \
    --input-file ../$DETOX_DEV_FILE \
    --output-file ../$VAL_FILE;

echo "Train: $TRAIN_FILE";
echo "Val: $VAL_FILE";
