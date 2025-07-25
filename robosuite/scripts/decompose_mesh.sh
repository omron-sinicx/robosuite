#!/bin/bash

MAX_CONVEX_HULL=30
THRESHOLD=0.01
SEED=42

usage() {
    echo "Usage: $0 -t <type> -n <name>"
    echo "  -t, --type: type of object (peg/hole)"
    echo "  -i, --input: input file name with .stl"
    echo "  -o, --output: output file name with .stl"
    echo "  -seed, --seed: random seed"
    echo "  -th, --threshold: convex threshold in [0.01, 1] (0.01: most fine-grained; 1: most coarse), default=$THRESHOLD"
    echo "  -max, --max-convex-hull: maximum number of convex hulls, default=$MAX_CONVEX_HULL (-1 for unlimited)"
    exit 1
}

[ $# -eq 0 ] && usage


while [ "$1" != "" ]; do
    case $1 in
        -h | --help)  usage ;;
        -t | --type)  TYPE=$2; shift ;;
        -i | --input)  INPUT=$2; shift ;;
        -o | --output)  OUTPUT=$2; shift ;;
        -seed | --seed)  SEED=$2; shift ;;
        -th | --threshold)  THRESHOLD=$2; shift ;;
        -max | --max-convex-hull)  MAX_CONVEX_HULL=$2; shift ;;
    esac
    shift
done

OUTPUT_WO_EXT="${OUTPUT%.*}"

if [ "$TYPE" = "peg" ]; then
    DEST="../models/assets/grippers/meshes/alan-char"
else
    DEST="../models/assets/objects/meshes/alan-char"
fi

# remove the old folder files to prevent contamination
if [ -d "./$OUTPUT_WO_EXT" ]; then
    rm -r ./$OUTPUT_WO_EXT/*
fi

python3 coacd_cb.py \
    -m multiple \
    --input $INPUT \
    --output $OUTPUT \
    --seed $SEED \
    --threshold $THRESHOLD \
    --max-convex-hull $MAX_CONVEX_HULL

# remove the old folder files to prevent contamination
if [ -d $DEST/$OUTPUT_WO_EXT ]; then
    rm -r $DEST/$OUTPUT_WO_EXT
fi

cp -r $OUTPUT_WO_EXT $DEST/