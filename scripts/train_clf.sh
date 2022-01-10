#!/bin/bash
set -e

TRAIN_FILE="data/clf_train.jsonl";
VAL_FILE="data/clf_val.jsonl";
TEST_FILE="data/clf_test.jsonl";


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

CUDA_VISIBLE_DEVICES=0 python3 -m clf.train \
    --train-path ../$TRAIN_FILE \
    --val-path ../$VAL_FILE \
    --test-path ../$TEST_FILE \
    --config-path ../$CONFIG_FILE \
    --out-dir ../$OUT_DIR;
