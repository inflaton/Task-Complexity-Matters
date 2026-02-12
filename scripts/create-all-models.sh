#!/bin/sh

BASEDIR=$(dirname "$0")
cd $BASEDIR/..
echo Current Directory:
pwd


./scripts/create-8k-model.sh deepseek-r1:8b

./scripts/create-8k-model.sh llama3.1:8b

./scripts/create-8k-model.sh deepseek-r1:14b

./scripts/create-8k-model.sh qwen2.5:14b

./scripts/create-8k-model.sh deepseek-r1:32b

./scripts/create-8k-model.sh qwen2.5:32b

./scripts/create-8k-model.sh deepseek-r1:70b

./scripts/create-8k-model.sh llama3.3:70b 


./scripts/create-16k-model.sh deepseek-r1:8b

./scripts/create-16k-model.sh llama3.1:8b

./scripts/create-16k-model.sh deepseek-r1:14b

./scripts/create-16k-model.sh qwen2.5:14b

./scripts/create-16k-model.sh deepseek-r1:32b

./scripts/create-16k-model.sh qwen2.5:32b

./scripts/create-16k-model.sh deepseek-r1:70b

./scripts/create-16k-model.sh llama3.3:70b 


./scripts/create-16k-model.sh qwen3:0.6b
./scripts/create-16k-model.sh qwen3:1.7b
./scripts/create-16k-model.sh qwen3:4b
./scripts/create-16k-model.sh qwen3:8b
./scripts/create-16k-model.sh qwen3:14b
./scripts/create-16k-model.sh qwen3:32b

./scripts/create-16k-model.sh granite3.3:2b
./scripts/create-16k-model.sh granite3.3:8b

./scripts/create-16k-model.sh magistral:24b
