#!/usr/bin/env bash
python trainval_net.py  --dataset pascal_voc \
                        --net vgg16 \
                        --bs 1 --nw 1 \
                        --epochs 6 \
                        --lr 0.001 \
                        --lr_decay_step 5 \
                        --cuda
