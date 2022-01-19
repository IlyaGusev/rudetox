#!/bin/bash
set -e

INPUT_PATH=$1
OUTPUT_PATH=$2
MARKER_MODEL="models/rubertconv_tokens"
FILLER_MODEL="models/rubert-base-cased-conversational"
TMP_INFERENCE_OUTPUT="predictions/condbert.jsonl"
TMP_RANKER_OUTPUT="predictions/condbert_ranker.jsonl"

cd rudetox;

CUDA_VISIBLE_DEVICE=0 python3 -m marker.infer_tokens_clf \
    --model-name "../$MARKER_MODEL" \
    --input-path "../$INPUT_PATH" \
    --output-path "../$TMP_INFERENCE_OUTPUT" \
    --filler-model-name "../$FILLER_MODEL" \
    --text-field "source";

CUDA_VISIBLE_DEVICES=0 python3 ranker.py \
    "../$TMP_INFERENCE_OUTPUT" \
    "../$TMP_RANKER_OUTPUT";

python3 to_plain.py \
    "../$TMP_RANKER_OUTPUT" \
    "../$OUTPUT_PATH";

echo $OUTPUT_PATH;
