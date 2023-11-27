#!/bin/bash
echo "Before calling cd, pwd=$PWD"
cd $PWD/src
echo "After calling cd, pwd=$PWD"

python3 main.py \
  --batch_size 1 \
  --epochs 20 \
  --lr 0.001 \
  --optimizer adam \
  --loss cross_entropy \
  --mode train \
