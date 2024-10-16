#!/bin/bash
batch_size=8

for batch_exp in {1..3}
do
    let batch_size*=2
    for num_workers in {0..3}
    do
        let num_workers*=2
        python3 benchmark_cv.py \
        --root $DATASET_ROOT/imagenette2-160/train \
        --batch_size $batch_size \
        --num_workers $num_workers \
        --model resnet50 \
        --n_iter 256
    done
done