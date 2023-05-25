#!/bin/bash
batch_size=1

for batch_exp in {1..5}
do
    let batch_size*=2
    for num_workers in {0..4}
    do
        python3 benchmark.py \
        --root /media/node_ale/DATA/datasets/celebrity-faces-dataset \
        --batch_size $batch_size \
        --num_workers $num_workers
    done
done