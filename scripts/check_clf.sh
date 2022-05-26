#!/bin/bash
set -e

usage() {
    echo "Usage: $0 -m MODEL_NAME -o OUT_FILE" 1>&2
}

exit_abnormal() {
    usage
    exit 1
}

m_flag=false;
o_flag=false;
s_flag=false;
while getopts ":m:o:s:" opt; do
  case $opt in
    m) MODEL_NAME="$OPTARG"; m_flag=true
    ;;
    o) OUT_FILE="$OPTARG"; o_flag=true
    ;;
    s) SAMPLE_RATE="$OPTARG"; s_flag=true
    ;;

    \?) echo "Invalid option -$OPTARG" >&2; exit_abnormal
    ;;
    :) echo "Missing option argument for -$OPTARG" >&2; exit_abnormal
    ;;
  esac
done

if ! $m_flag
then
    echo "Missing -m option (model name)"; exit_abnormal;
fi

if ! $o_flag
then
    echo "Missing -o option (path to output dir)"; exit_abnormal;
fi

if ! $s_flag
then
    SAMPLE_RATE=0.1;
fi



TEST_FILE="data/final/clf_test.jsonl";
VOCAB_FILE="data/bad_vocab.txt";

CUDA_VISIBLE_DEVICES=0 python3.9 -m rudetox.clf.check \
    --model-name $MODEL_NAME \
    --test-path $TEST_FILE \
    --toxic-vocab-path $VOCAB_FILE \
    --sample-rate $SAMPLE_RATE \
    --save-path $OUT_FILE \
    --manual-test-path data/custom_test.jsonl;
