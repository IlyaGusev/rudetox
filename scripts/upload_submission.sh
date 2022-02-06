#!/bin/bash
set -e

INPUT_PATH=$1;

python3 -m rudetox.to_jsonl $INPUT_PATH data/detox_test.tsv $INPUT_PATH.style style;
python3 -m rudetox.to_jsonl $INPUT_PATH data/detox_test.tsv $INPUT_PATH.sim sim;

python3 -m rudetox.crowd.upload \
    --input-path $INPUT_PATH.style \
    --honey-path toloka/style/examples/honey.tsv \
    --template-pool-id 31216701 \
    --key-fields text \
    --input-fields text \
    --name $INPUT_PATH;
python3 -m rudetox.crowd.upload \
    --input-path $INPUT_PATH.sim \
    --honey-path toloka/similarity/examples/honey.tsv \
    --template-pool-id 31251898 \
    --key-fields first_text,second_text \
    --input-fields first_text,second_text \
    --name $INPUT_PATH;
