#!/bin/sh

BASEDIR=$(dirname "$0")
cd $BASEDIR/..
echo Current Directory:
pwd

MODEL=$1

multiline_string=$(cat <<EOF
FROM $MODEL
PARAMETER num_ctx 32768
EOF
)

echo "$multiline_string" > temp.txt

sudo ollama create ${MODEL}_32k -f temp.txt

rm temp.txt
