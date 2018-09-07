#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 \
python predict.py --img_file I00088.jpg \
                  --net vgg16 \
                  --cuda --vis