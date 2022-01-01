#!/bin/bash

DETOX_INPUT_FILE="../data/russe_detox_2022/train.tsv";
OK_INPUT_FILE="../data/ok.ft";
CH_INPUT_FILE="../data/2ch.csv";
VOCAB_FILE="../data/bad_vocab.txt";

OUTPUT_FILE="../data/clf_all.jsonl";
TRAIN_FILE="../data/clf_train.jsonl";
VAL_FILE="../data/clf_val.jsonl";
TEST_FILE="../data/clf_test.jsonl";

DETOX_TEMP_FILE=$(mktemp);
OK_TEMP_FILE=$(mktemp);
CH_TEMP_FILE=$(mktemp);
OUTPUT_TEMP_FILE=$(mktemp);

python3.9 -m clf.converters.detox_to_clf_jsonl \
    --input-file $DETOX_INPUT_FILE \
    --output-file $DETOX_TEMP_FILE;
python3.9 -m clf.converters.ok_to_clf_jsonl \
    --input-file $OK_INPUT_FILE \
    --output-file $OK_TEMP_FILE;
python3.9 -m clf.converters.2ch_to_clf_jsonl \
    --input-file $CH_INPUT_FILE \
    --output-file $CH_TEMP_FILE;
python3.9 -m clf.merge_all \
    $DETOX_TEMP_FILE $OK_TEMP_FILE $CH_TEMP_FILE \
    --output-file $OUTPUT_TEMP_FILE;
python3.9 -m clf.clean \
    --input-path $OUTPUT_TEMP_FILE \
    --output-path $OUTPUT_FILE \
    --bad-vocab-path $VOCAB_FILE;
python3.9 -m clf.split \
    --input-path $OUTPUT_FILE \
    --train-path $TRAIN_FILE \
    --val-path $VAL_FILE \
    --test-path $TEST_FILE;

echo $OUTPUT_FILE;

rm -f $DETOX_TEMP_FILE $OK_TEMP_FILE $CH_TEMP_FILE $OUTPUT_TEMP_FILE;
