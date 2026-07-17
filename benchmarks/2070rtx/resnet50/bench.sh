#!/bin/bash
batch_size=8

for batch_exp in {1..3}
do
    let batch_size*=2
    for num_workers in {0..3}
    do
        set -x

        let num_workers*=2
        python3 benchmark_cv.py \
        --root $DATASET_ROOT/imagewoof2-160/train \
        --batch_size $batch_size \
        --num_workers $num_workers \
        --model resnet50 \
        --num_iters 64 \
        --img_size 224

        set +x
    done
done


batch_size=8

for batch_exp in {1..3}
do
    let batch_size*=2
    for num_workers in {0..3}
    do
        let num_workers*=2
        python3 benchmark_cv.py \
        --root $DATASET_ROOT/imagewoof2-160/train \
        --batch_size $batch_size \
        --num_workers $num_workers \
        --model resnet50 \
        --num_iters 64 \
        --eval
    done
done
