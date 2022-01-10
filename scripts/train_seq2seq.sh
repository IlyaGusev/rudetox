#!/bin/bash
set -e

TRAIN_FILE="data/seq2seq_train.jsonl" \
VAL_FILE="data/seq2seq_val.jsonl";


usage() {
    echo "Usage: $0 -c CONFIG_PATH -o OUT_DIR" 1>&2
}

exit_abnormal() {
    usage
    exit 1
}

c_flag=false;
o_flag=false;
while getopts ":c:o:" opt; do
  case $opt in
    c) CONFIG_FILE="$OPTARG"; c_flag=true
    ;;
    o) OUT_DIR="$OPTARG"; o_flag=true
    ;;

    \?) echo "Invalid option -$OPTARG" >&2; exit_abnormal
    ;;
    :) echo "Missing option argument for -$OPTARG" >&2; exit_abnormal
    ;;
  esac
done

if ! $c_flag
then
    echo "Missing -c option (path to config)"; exit_abnormal;
fi

if ! $o_flag
then
    echo "Missing -o option (path to output dir)"; exit_abnormal;
fi

mkdir -p models
cd rudetox;

CUDA_VISIBLE_DEVICES=0 python3 -m seq2seq.train \
  --config-file ../$CONFIG_FILE \
  --train-file ../$TRAIN_FILE \
  --val-file ../$VAL_FILE \
  --output-dir ../$OUT_DIR \
  --source-field "source" \
  --target-field "target";
