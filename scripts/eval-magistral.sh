#!/bin/sh

BASEDIR=$(dirname "$0")
cd $BASEDIR/..
echo Current Directory:
pwd

export DATA_PATH=dataset/amazon_reviews.csv
# export OUTPUT_COLUMN="Review-sentiment"

# export THINKING=0
# export RESULTS_CSV=results/magistral-amazon_no_thinking.csv

# ./scripts/eval-ollama-model.sh magistral:24b_16k $RESULTS_CSV


# export THINKING=1
# export RESULTS_CSV=results/magistral-amazon_thinking.csv

# ./scripts/eval-ollama-model.sh magistral:24b_16k $RESULTS_CSV


export START_NUM_SHOTS=0
export END_NUM_SHOTS=0

export DATA_PATH=dataset/imdb_reviews.csv
export OUTPUT_COLUMN="Review-basic-sentiment"

export THINKING=0
export RESULTS_CSV=results/magistral-imdb_no_thinking.csv

./scripts/eval-ollama-model.sh magistral:24b_16k $RESULTS_CSV


export THINKING=1
export RESULTS_CSV=results/magistral-imdb_thinking.csv

./scripts/eval-ollama-model.sh magistral:24b_16k $RESULTS_CSV


export DATA_PATH=dataset/GoEmotions.csv
export OUTPUT_COLUMN="Emotion"

export THINKING=0
export RESULTS_CSV=results/magistral-goemotions_no_thinking.csv

./scripts/eval-ollama-model.sh magistral:24b_16k $RESULTS_CSV


export THINKING=1
export RESULTS_CSV=results/magistral-goemotions_thinking.csv

./scripts/eval-ollama-model.sh magistral:24b_16k $RESULTS_CSV




export DATA_PATH=dataset/imdb_reviews.csv
export OUTPUT_COLUMN="Review-basic-sentiment"

export THINKING=0
export RESULTS_CSV=results/granite-imdb_no_thinking.csv

./scripts/eval-ollama-model.sh granite3.3:2b_16k $RESULTS_CSV
# ./scripts/eval-ollama-model.sh granite3.3:8b_16k $RESULTS_CSV

export THINKING=1
export RESULTS_CSV=results/granite-imdb_thinking.csv

./scripts/eval-ollama-model.sh granite3.3:2b_16k $RESULTS_CSV
# ./scripts/eval-ollama-model.sh granite3.3:8b_16k $RESULTS_CSV

export DATA_PATH=dataset/GoEmotions.csv
export OUTPUT_COLUMN="Emotion"

export THINKING=0
export RESULTS_CSV=results/granite-goemotions_no_thinking.csv

./scripts/eval-ollama-model.sh granite3.3:2b_16k $RESULTS_CSV
# ./scripts/eval-ollama-model.sh granite3.3:8b_16k $RESULTS_CSV

export THINKING=1
export RESULTS_CSV=results/granite-goemotions_thinking.csv

./scripts/eval-ollama-model.sh granite3.3:2b_16k $RESULTS_CSV
# ./scripts/eval-ollama-model.sh granite3.3:8b_16k $RESULTS_CSV