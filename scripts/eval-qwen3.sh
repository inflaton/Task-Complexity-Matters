#!/bin/sh

BASEDIR=$(dirname "$0")
cd $BASEDIR/..
echo Current Directory:
pwd

export DATA_PATH=dataset/amazon_reviews.csv
export OUTPUT_COLUMN="Review-sentiment"

export THINKING=0
export RESULTS_CSV=results/qwen3-amazon_no_thinking.csv

./scripts/eval-ollama-model.sh qwen3:4b_16k $RESULTS_CSV
./scripts/eval-ollama-model.sh qwen3:8b_16k $RESULTS_CSV
./scripts/eval-ollama-model.sh qwen3:14b_16k $RESULTS_CSV
./scripts/eval-ollama-model.sh qwen3:32b_16k $RESULTS_CSV

export THINKING=1
export RESULTS_CSV=results/qwen3-amazon_thinking.csv

./scripts/eval-ollama-model.sh qwen3:4b_16k $RESULTS_CSV
./scripts/eval-ollama-model.sh qwen3:8b_16k $RESULTS_CSV
./scripts/eval-ollama-model.sh qwen3:14b_16k $RESULTS_CSV
./scripts/eval-ollama-model.sh qwen3:32b_16k $RESULTS_CSV

export DATA_PATH=dataset/imdb_reviews.csv
export OUTPUT_COLUMN="Review-basic-sentiment"

export THINKING=0
export RESULTS_CSV=results/qwen3-imdb_no_thinking.csv

./scripts/eval-ollama-model.sh qwen3:4b_16k $RESULTS_CSV
./scripts/eval-ollama-model.sh qwen3:8b_16k $RESULTS_CSV
./scripts/eval-ollama-model.sh qwen3:14b_16k $RESULTS_CSV
./scripts/eval-ollama-model.sh qwen3:32b_16k $RESULTS_CSV

export THINKING=1
export RESULTS_CSV=results/qwen3-imdb_thinking.csv

./scripts/eval-ollama-model.sh qwen3:4b_16k $RESULTS_CSV
./scripts/eval-ollama-model.sh qwen3:8b_16k $RESULTS_CSV
./scripts/eval-ollama-model.sh qwen3:14b_16k $RESULTS_CSV
./scripts/eval-ollama-model.sh qwen3:32b_16k $RESULTS_CSV

export DATA_PATH=dataset/GoEmotions.csv
export OUTPUT_COLUMN="Emotion"

export THINKING=0
export RESULTS_CSV=results/qwen3-goemotions_no_thinking.csv

./scripts/eval-ollama-model.sh qwen3:4b_16k $RESULTS_CSV
./scripts/eval-ollama-model.sh qwen3:8b_16k $RESULTS_CSV
./scripts/eval-ollama-model.sh qwen3:14b_16k $RESULTS_CSV
./scripts/eval-ollama-model.sh qwen3:32b_16k $RESULTS_CSV

export THINKING=1
export RESULTS_CSV=results/qwen3-goemotions_thinking.csv

./scripts/eval-ollama-model.sh qwen3:4b_16k $RESULTS_CSV
./scripts/eval-ollama-model.sh qwen3:8b_16k $RESULTS_CSV
./scripts/eval-ollama-model.sh qwen3:14b_16k $RESULTS_CSV
./scripts/eval-ollama-model.sh qwen3:32b_16k $RESULTS_CSV

