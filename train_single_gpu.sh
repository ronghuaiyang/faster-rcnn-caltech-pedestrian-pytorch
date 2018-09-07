#!/usr/bin/env bash
GPU_ID=$1
CUDA_VISIBLE_DEVICES=$GPU_ID \
python trainval_net.py  --dataset caltech \
                        --net res50 \
                        --bs 1 --nw 1 \
                        --lr 0.001 \
                        --epochs 6 \
                        --lr_decay_step 5 \
                        --cuda
