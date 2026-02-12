#!/bin/sh

BASEDIR=$(dirname "$0")
cd $BASEDIR/..
echo Current Directory:
pwd

export NUM_CTX=8192
export RESULTS_PATH=results/amazon_reviews_results_small.csv

./scripts/eval-model.sh deepseek-r1:8b_8k

./scripts/eval-model.sh llama3.1:8b_8k

./scripts/eval-model.sh deepseek-r1:14b_8k

./scripts/eval-model.sh qwen2.5:14b_8k
