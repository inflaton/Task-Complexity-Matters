#!/bin/sh

BASEDIR=$(dirname "$0")
cd $BASEDIR/..
echo Current Directory:
pwd

export NUM_CTX=8192
export DATA_PATH=dataset/imdb_reviews.csv
export RESULTS_PATH=results/imdb_reviews_results_all.csv
export OUTPUT_COLUMN="Review-basic-sentiment"

# ./scripts/eval-model.sh deepseek-r1:8b_32k

# ./scripts/eval-model.sh deepseek-r1:14b_32k

# ./scripts/eval-model.sh deepseek-r1:32b_32k

# ./scripts/eval-model.sh deepseek-r1:70b_32k

./scripts/eval-model.sh llama3.1:8b_32k

./scripts/eval-model.sh qwen2.5:14b_32k

./scripts/eval-model.sh qwen2.5:32b_32k

./scripts/eval-model.sh llama3.3:70b_32k 
