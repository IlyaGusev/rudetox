#!/bin/bash

DETOX_INPUT_FILE="data/detox_train.tsv";
OK_INPUT_FILE="data/ok.ft";
CH_INPUT_FILE="data/2ch.csv";
PERSONA_INPUT_FILE="data/persona.tsv";
KOZIEV_INPUT_FILE="data/koziev.txt"
VOCAB_FILE="data/bad_vocab.txt";

OUTPUT_FILE="data/clf_all.jsonl";
TRAIN_FILE="data/clf_train.jsonl";
VAL_FILE="data/clf_val.jsonl";
TEST_FILE="data/clf_test.jsonl";

DETOX_TEMP_FILE=$(mktemp);
OK_TEMP_FILE=$(mktemp);
CH_TEMP_FILE=$(mktemp);
OUTPUT_TEMP_FILE=$(mktemp);
PERSONA_TEMP_FILE=$(mktemp);
KOZIEV_TEMP_FILE=$(mktemp);

cd scripts;

echo "Detox processing...";
python3 -m clf.converters.detox \
    --input-file ../$DETOX_INPUT_FILE \
    --output-file $DETOX_TEMP_FILE;
echo "OK processing...";
python3 -m clf.converters.ok \
    --input-file ../$OK_INPUT_FILE \
    --output-file $OK_TEMP_FILE;
echo "2ch processing...";
python3 -m clf.converters.2ch \
    --input-file ../$CH_INPUT_FILE \
    --output-file $CH_TEMP_FILE;
echo "Persona processing...";
python3 -m clf.converters.persona \
    --input-file ../$PERSONA_INPUT_FILE \
    --output-file $PERSONA_TEMP_FILE;
echo "Koziev processing...";
python3 -m clf.converters.koziev \
    --input-file ../$KOZIEV_INPUT_FILE \
    --output-file $KOZIEV_TEMP_FILE \
    --vocab-file ../$VOCAB_FILE;

echo "Merging...";
python3 -m clf.merge_all \
    $DETOX_TEMP_FILE $OK_TEMP_FILE $CH_TEMP_FILE $PERSONA_TEMP_FILE $KOZIEV_TEMP_FILE \
    --output-file $OUTPUT_TEMP_FILE;
echo "Cleaning...";
python3 -m clf.clean \
    --input-path $OUTPUT_TEMP_FILE \
    --output-path ../$OUTPUT_FILE \
    --bad-vocab-path ../$VOCAB_FILE;
echo "Split...";
python3 -m clf.split \
    --input-path ../$OUTPUT_FILE \
    --train-path ../$TRAIN_FILE \
    --val-path ../$VAL_FILE \
    --test-path ../$TEST_FILE;

echo "All: $OUTPUT_FILE";
echo "Train: $TRAIN_FILE";
echo "Val: $VAL_FILE";
echo "Test: $TEST_FILE";

rm -f $DETOX_TEMP_FILE $OK_TEMP_FILE $CH_TEMP_FILE $OUTPUT_TEMP_FILE $PERSONA_TEMP_FILE $KOZIEV_TEMP_FILE;