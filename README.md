# A *Faster* Pytorch Implementation for Caltech Pedestrian dataset

This implement is based on [jwyang/faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch). I make some changes to adapt the pedestrian detection.

## References

* [jwyang/faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch)

* [ruotianluo/pytorch-resnet](https://github.com/ruotianluo/pytorch-resnet)

* [GBJim/mscnn](https://github.com/GBJim/mscnn)

* [ChaoPei/faster-rcnn-pedestrian-detection](https://github.com/ChaoPei/faster-rcnn-pedestrian-detection)

* [CasiaFan/Dataset_to_VOC_converter](https://github.com/CasiaFan/Dataset_to_VOC_converter)

## Tutorial
* use [CasiaFan/Dataset_to_VOC_converter](https://github.com/CasiaFan/Dataset_to_VOC_converter) to create training data

* build modules

```Shell
cd lib
sh make.sh
```

* train_single_gpu.sh for training on single gpu
```Shell
sh train_single_gpu.sh 0
```

* train_mgpu.sh for trainning on multi-gpus
```Shell
sh train_mgpus.sh
```

* predict single image
```Shell
sh predict.sh
```

* create caltech eval result
```Shell
sh caltech_dataset_detect.sh
```