#!/bin/bash

# Dir of this script
SCRIPT_DIR=$(dirname "$0")

# Check if .env file exists
ENV_FILE="$SCRIPT_DIR/../.env"
if [ ! -f "$ENV_FILE" ]; then
    echo "Error: .env file does not exist."
    exit 1
fi

# Load Environment Variables
export $(cat "$ENV_FILE" | xargs)

# Check if ROOT_PATH is set
if [ -z "${ROOT_PATH}" ]; then
    echo "Error: ROOT_PATH is not set in .env file."
    exit 1
fi
echo "ROOT_PATH is set to '$ROOT_PATH'"

# Move the working directory to ROOT_PATH
cd $ROOT_PATH

python3 -m src.main \
  --batch_size 1 \
  --epochs 20 \
  --learning_rate 0.001 \
  --optimizer adam \
  --loss cross_entropy \
  --num_samples 900 \
  --mode train \
