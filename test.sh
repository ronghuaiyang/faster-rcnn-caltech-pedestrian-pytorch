 CUDA_VISIBLE_DEVICES=0 python test_net.py --dataset caltech \
                   --net vgg16 \
                   --checksession 1 --checkepoch 6 --checkpoint 55243 \
                   --cuda
