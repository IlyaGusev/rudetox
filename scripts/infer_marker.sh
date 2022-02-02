#!/bin/bash
set -e

INPUT_PATH=$1
OUTPUT_PATH=$2
SAMPLE_RATE=0.001
TEXT_FIELD="text"
MODEL_TYPE="mlm"
MASK_TOKEN="[MASK]"
#MODEL_TYPE="seq2seq"
#MASK_TOKEN="<extra_id_0>"

MARKER_MODEL="models/rubertconv_toxic_marker"
FILLER_MODEL="models/rubert-base-cased-conversational"
#FILLER_MODEL="models/mt5-small"
TMP_INFERENCE_OUTPUT="predictions/rubertconv.jsonl"
#TMP_INFERENCE_OUTPUT="predictions/mt5_small.jsonl"


CUDA_VISIBLE_DEVICE=0 python3 -m rudetox.marker.fill \
    --model-name "$MARKER_MODEL" \
    --input-path "$INPUT_PATH" \
    --output-path "$TMP_INFERENCE_OUTPUT" \
    --filler-model-name "$FILLER_MODEL" \
    --text-field $TEXT_FIELD \
    --filler-model-type $MODEL_TYPE \
    --filler-mask-token $MASK_TOKEN \
    --sample-rate $SAMPLE_RATE;

CUDA_VISIBLE_DEVICES=0 python3 -m rudetox.ranker \
    "$TMP_INFERENCE_OUTPUT" \
    "$OUTPUT_PATH";

echo $OUTPUT_PATH;
