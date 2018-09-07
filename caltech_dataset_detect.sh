#!/usr/bin/env bash
GPU_ID=$1
CUDA_VISIBLE_DEVICES=$GPU_ID \
python caltech_dataset_detection.py --net vgg16 \
                                    --detection fasterrcnn \
                                    --cuda