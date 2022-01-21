#!/bin/bash
set -e

DETOX_TRAIN_FILE="data/detox_train.tsv";
DETOX_DEV_FILE="data/detox_dev.tsv";

GEN_FILE="data/seq2seq_gen.jsonl"
PRE_TRAIN_FILE="data/seq2seq_gen_train.jsonl"
PRE_VAL_FILE="data/seq2seq_gen_val.jsonl"
TRAIN_FILE="data/seq2seq_train.jsonl";
VAL_FILE="data/seq2seq_val.jsonl";
TEST_FILE="data/seq2seq_test.jsonl";
UNITED_TRAIN_FILE="data/seq2seq_united_train.jsonl";

cd rudetox;

echo "Detox processing...";
python3 -m seq2seq.converters.detox \
    --input-file ../$DETOX_TRAIN_FILE \
    --output-file ../$TRAIN_FILE;

python3 -m seq2seq.converters.detox \
    --input-file ../$DETOX_DEV_FILE \
    --output-file ../$VAL_FILE;

python3 -m seq2seq.converters.detox \
    --input-file ../$DETOX_DEV_FILE \
    --output-file ../$TEST_FILE \
    --source-only;

cat ../$GEN_FILE > ../$UNITED_TRAIN_FILE;
cat ../$TRAIN_FILE >> ../$UNITED_TRAIN_FILE;
shuf ../$UNITED_TRAIN_FILE > ../$UNITED_TRAIN_FILE.shuf;
mv ../$UNITED_TRAIN_FILE.shuf ../$UNITED_TRAIN_FILE;

echo "Pretrain Train: $PRE_TRAIN_FILE"
echo "Pretrain Val: $PRE_VAL_FILE"
echo "Train: $TRAIN_FILE";
echo "Val: $VAL_FILE";
echo "United Train: $UNITED_TRAIN_FILE"
echo "Test: $TEST_FILE";
