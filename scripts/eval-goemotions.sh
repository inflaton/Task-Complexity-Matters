#!/bin/sh

BASEDIR=$(dirname "$0")
cd $BASEDIR/..
echo Current Directory:
pwd

export NUM_CTX=8192
export DATA_PATH=dataset/GoEmotions.csv
export RESULTS_PATH=results/GoEmotions_results.csv
export OUTPUT_COLUMN="Emotion"

# ./scripts/eval-model.sh deepseek-r1:8b_8k

#./scripts/eval-model.sh deepseek-r1:14b_8k

#./scripts/eval-model.sh deepseek-r1:32b_8k

#./scripts/eval-model.sh deepseek-r1:70b_8k

./scripts/eval-model.sh llama3.1:8b_8k

./scripts/eval-model.sh qwen2.5:14b_8k

./scripts/eval-model.sh qwen2.5:32b_8k

./scripts/eval-model.sh llama3.3:70b_8k 