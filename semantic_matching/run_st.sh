#!/bin/bash
set -e

# text-text
for dataset in MRPC MultiNLI quora SciTail SNLI
do
    for seed in 0 1 2
    do
        CUDA_VISIBLE_DEVICES=1 python st_t2t.py --seed $seed --dataname $dataset
    done
done

# image-image
for dataset in airbnb cub200 iPanda sea_turtle SOP
do
    for seed in 0 1 2
    do
        CUDA_VISIBLE_DEVICES=1 python st_i2i.py --seed $seed --dataname $dataset
    done
done

# image-text
for dataset in cub200 flickr30k flower102 ImageParagraphCap mscoco
do
    for seed in 0 1 2
    do
        CUDA_VISIBLE_DEVICES=1 python st_i2t.py --seed $seed --dataname $dataset
    done
done