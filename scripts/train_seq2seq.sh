#!/bin/bash
set -e

usage() {
    echo "Usage: $0 -t TRAIN_PATH -v VAL_PATH -c CONFIG_PATH -o OUT_DIR" 1>&2
}

exit_abnormal() {
    usage
    exit 1
}

t_flag=false;
v_flag=false;
c_flag=false;
o_flag=false;
while getopts ":c:o:t:v:" opt; do
  case $opt in
    c) CONFIG_FILE="$OPTARG"; c_flag=true
    ;;
    o) OUT_DIR="$OPTARG"; o_flag=true
    ;;
    t) TRAIN_FILE="$OPTARG"; t_flag=true
    ;;
    v) VAL_FILE="$OPTARG"; v_flag=true
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

if ! $t_flag
then
    echo "Missing -t option (path to train)"; exit_abnormal;
fi

if ! $v_flag
then
    echo "Missing -v option (path to val)"; exit_abnormal;
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
