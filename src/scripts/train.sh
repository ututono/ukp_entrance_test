#!/bin/bash


python3 src/main.py \
  --batch_size 1 \
  --epochs 20 \
  --lr 0.001 \
  --optimizer adam \
  --loss cross_entropy \
  --mode train \
