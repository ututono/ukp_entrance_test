#!/bin/bash

cd /home/yu_z@PTW.Maschinenbau.TU-Darmstadt.de/entrance_test/

python3 -m src.main \
  --batch_size 1 \
  --epochs 20 \
  --learning_rate 0.001 \
  --optimizer adam \
  --loss cross_entropy \
  --mode train \
