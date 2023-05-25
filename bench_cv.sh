#!/bin/bash
batch_size=2

for batch_exp in {1..5}
do
    let batch_size*=2
    for num_workers in {0..4}
    do
        python3 benchmark.py \
        --root $DATASET_ROOT/GPR1200/images \
        --batch_size $batch_size \
        --num_workers $num_workers \
        --pin_memory
    done
done