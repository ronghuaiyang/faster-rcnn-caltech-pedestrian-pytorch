#!/usr/bin/env bash
python trainval_net.py  --dataset pascal_voc \
                        --net vgg16 \
                        --bs 8 --nw 8 \
                        --epochs 80 \
                        --lr 0.001 \
                        --lr_decay_step 30 \
                        --cuda --mGPUs

# python test_net.py --dataset pascal_voc \
#                    --net vgg16 \
#                    --checksession 1 --checkepoch 80 --checkpoint 2504 \
#                    --cuda
