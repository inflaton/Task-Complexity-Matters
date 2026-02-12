#!/bin/sh

BASEDIR=$(dirname "$0")
cd $BASEDIR/..
echo Current Directory:
pwd

export MODEL_NAME=$1
echo Evaluating $MODEL_NAME

ollama run $MODEL_NAME 'hi'
ollama ps

python llm_toolkit/eval_openai.py

ollama stop $MODEL_NAME
