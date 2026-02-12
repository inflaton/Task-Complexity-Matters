#!/bin/sh

BASEDIR=$(dirname "$0")
cd $BASEDIR/..
echo Current Directory:
pwd

export MODEL_NAME=$1

if [ -z "$MODEL_NAME" ]; then
  echo "Usage: $0 <model_name> <results_path>"
  exit 1
fi

export RESULTS_PATH=$2

if [ -z "$RESULTS_PATH" ]; then
  echo "Usage: $0 <model_name> <results_path>"
  exit 1
fi

echo "Evaluating model: $MODEL_NAME"
echo "Results Path: $RESULTS_PATH"
echo "Running evaluation script..."
python llm_toolkit/eval_openai.py