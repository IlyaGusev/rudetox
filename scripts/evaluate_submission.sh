#!/bin/bash
set -e

STYLE_POOL=$1;
SIM_POOL=$2;

python3 -m rudetox.crowd.aggregate \
    --input-fields text \
    --agg-output data/style_agg.txt \
    --raw-output data/style_raw.txt \
    --pools $STYLE_POOL \
    --key-fields text;
python3 -m rudetox.crowd.aggregate \
    --input-fields first_text,second_text \
    --agg-output data/sim_agg.txt \
    --raw-output data/sim_raw.txt\
    --pools $SIM_POOL \
    --key-fields first_text,second_text;
