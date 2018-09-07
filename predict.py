from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import pickle
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.blob import prep_im_for_blob
from model.rpn.bbox_transform import clip_boxes
from model.nms.nms_wrapper import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='predict image')
    parser.add_argument('--img_file', dest='img_file',
                        help='input image file',
                        default='test.jpg', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152',
                        default='res101', type=str)
    parser.add_argument('--load_name', dest='load_name',
                        help='load model name', default="./models/vgg16/caltech/faster_rcnn_1_6_55243.pth",
                        type=str)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--vis', dest='vis',
                        help='visualization mode',
                        action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()

    # pedestrian anchor box setting
    args.cfg_file = "cfgs/{}.yml".format(args.net)
    cfg.ANCHOR_SCALES = 2.5*(1.4**np.arange(0, 9))
    cfg.ANCHOR_RATIOS = [1/0.4]

    print('Called with args:')
    print(args)

    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    print('Using config:')
    pprint.pprint(cfg)

    # initilize the network here.
    classes = ('__background__', 'person')
    if args.net == 'vgg16':
        fasterRCNN = vgg16(classes, pretrained=False, class_agnostic=False)
    elif args.net == 'res101':
        fasterRCNN = resnet(classes, 101, pretrained=False, class_agnostic=False)
    elif args.net == 'res50':
        fasterRCNN = resnet(classes, 50, pretrained=False, class_agnostic=False)
    elif args.net == 'res152':
        fasterRCNN = resnet(classes, 152, pretrained=False, class_agnostic=False)
    else:
        print("network is not defined")

    fasterRCNN.create_architecture()
    fasterRCNN.eval()

    load_name = args.load_name
    print("load checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    fasterRCNN.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']

    # prepare input data
    img_file = args.img_file
    img = cv2.imread(img_file)
    if img is None:
        print("read img_file error!")
        sys.exit()

    target_size = cfg.TRAIN.SCALES[0]
    im, im_scale = prep_im_for_blob(img, cfg.PIXEL_MEANS, target_size, cfg.TRAIN.MAX_SIZE)
    im = im.transpose((2, 0, 1))
    im_info = np.array([[im.shape[1], im.shape[2], im_scale]], dtype=np.float32)
    im = im[np.newaxis, :]
    im_data = torch.from_numpy(im)
    im_info = torch.from_numpy(im_info)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    if args.cuda:
        fasterRCNN.cuda()
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        gt_boxes = gt_boxes.cuda()
        num_boxes = num_boxes.cuda()

    # test settings
    num_classes = len(classes)
    max_per_image = 100
    vis = args.vis
    if vis:
        thresh = 0.05
    else:
        thresh = 0.0
    all_boxes = [[] for _ in xrange(num_classes)]

    # inference
    start = time.time()
    det_tic = time.time()

    output = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)
    rois = output[0]
    cls_prob = output[1]
    bbox_pred = output[2]
    
    scores = cls_prob.data
    boxes = rois.data[:, :, 1:5]
    
    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = bbox_pred.data
        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                         + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
            box_deltas = box_deltas.view(1, -1, 4 * num_classes)

        pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
        pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    pred_boxes /= im_info[0][2].cuda()

    scores = scores.squeeze()
    pred_boxes = pred_boxes.squeeze()

    det_toc = time.time()
    detect_time = det_toc - det_tic
    misc_tic = time.time()
    if vis:
        im = cv2.imread(img_file)
        im2show = np.copy(im)
    for j in xrange(1, num_classes):
        inds = torch.nonzero(scores[:, j] > thresh).view(-1)
        # if there is det
        if inds.numel() > 0:
            cls_scores = scores[:, j][inds]
            _, order = torch.sort(cls_scores, 0, True)
            cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
            cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
            cls_dets = cls_dets[order]
            keep = nms(cls_dets, cfg.TEST.NMS, not cfg.USE_GPU_NMS)
            cls_dets = cls_dets[keep.view(-1).long()]
            if vis:
                im2show = vis_detections(im2show, classes[j], cls_dets.cpu().numpy(), 0.3)
            all_boxes[j] = cls_dets.cpu().numpy()
        else:
            all_boxes[j] = np.transpose(np.array([[], [], [], [], []]), (1, 0))

    # Limit to max_per_image detections *over all classes*
    if max_per_image > 0:
        image_scores = np.hstack([all_boxes[j][:, -1] for j in xrange(1, num_classes)])
        if len(image_scores) > max_per_image:
            image_thresh = np.sort(image_scores)[-max_per_image]
            for j in xrange(1, num_classes):
                keep = np.where(all_boxes[j][:, -1] >= image_thresh)[0]
                all_boxes[j] = all_boxes[j][keep, :]

    misc_toc = time.time()
    nms_time = misc_toc - misc_tic

    end = time.time()

    if vis:
        cv2.imwrite('result.png', im2show)
        cv2.imshow('test', im2show)
        cv2.waitKey(0)

    print("detect time: %0.4fs" % detect_time)
    print("nms time: %0.4fs" % nms_time)
    print("test time: %0.4fs" % (end - start))


