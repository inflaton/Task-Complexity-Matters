#!/bin/sh

BASEDIR=$(dirname "$0")
cd $BASEDIR/..
echo Current Directory:
pwd

# Qwen3 family (4B, 8B, 14B, 32B × thinking/non-thinking × 3 datasets)
bash scripts/eval-qwen3.sh

# Granite3.3 family (2B, 8B × thinking/non-thinking × 3 datasets)
bash scripts/eval-granite.sh

# Magistral (24B × thinking/non-thinking × 3 datasets)
bash scripts/eval-magistral.sh
