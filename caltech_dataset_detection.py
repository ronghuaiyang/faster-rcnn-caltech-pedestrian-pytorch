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
import glob
import re
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


CALTECH_DATA_PATH = "/data1/Datasets/caltech_pedestrian/caltech_convert/data"
IMG_PATH = os.path.join(CALTECH_DATA_PATH, "images")
CLASSES = ('__background__', 'person')
OUTPUT_PATH = "/data1/Datasets/caltech_pedestrian/code3.2.1/data-USA/res"


class Timer(object):
    """A simple timer."""

    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='predict image')

    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152',
                        default='res101', type=str)
    parser.add_argument('--load_name', dest='load_name',
                        help='load model name', default="./models/vgg16/caltech/faster_rcnn_1_6_55243.pth",
                        type=str)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--detection', dest='dt_name',
                        help='model to test', default='detection_1',
                        type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


# def filter_proposals(proposals, threshold=-10):
#     # Bug 1 Fixed
#     keeps = (proposals[:, -1] >= threshold) & (proposals[:, 2] != 0) & (proposals[:, 3] != 0)
#
#     return keeps


def im_detect(net, file_path):
    im = cv2.imread(file_path)
    im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, cfg.TEST.SCALES[0], cfg.TRAIN.MAX_SIZE)
    im = im.transpose((2, 0, 1))
    im_info = np.array([[im.shape[1], im.shape[2], im_scale]], dtype=np.float32)
    im = im[np.newaxis, :]
    im_data = torch.from_numpy(im)
    im_info = torch.from_numpy(im_info)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    if args.cuda:
        net.cuda()
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        gt_boxes = gt_boxes.cuda()
        num_boxes = num_boxes.cuda()

    # test settings
    num_classes = len(CLASSES)
    max_per_image = 100
    all_boxes = [[] for _ in xrange(num_classes)]

    # inference
    output = net(im_data, im_info, gt_boxes, num_boxes)
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

    for j in xrange(1, num_classes):
        inds = torch.nonzero(scores[:, j] > 0).view(-1)
        # if there is det
        if inds.numel() > 0:
            cls_scores = scores[:, j][inds]
            _, order = torch.sort(cls_scores, 0, True)
            cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
            cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
            cls_dets = cls_dets[order]
            keep = nms(cls_dets, cfg.TEST.NMS, not cfg.USE_GPU_NMS)
            cls_dets = cls_dets[keep.view(-1).long()]
            # im2show = vis_detections(im2show, CLASSES[j], cls_dets.cpu().numpy(), 0.01)
            all_boxes[j] = cls_dets.cpu().numpy()
        else:
            all_boxes[j] = np.transpose(np.array([[], [], [], [], []]), (1, 0))

    return all_boxes[1]


def write_caltech_results_file(net):
    # The follwing nested fucntions are for smart sorting
    def atoi(text):
        return int(text) if text.isdigit() else text

    def natural_keys(text):
        return [atoi(c) for c in re.split('(\d+)', text)]

    def insert_frame(target_frames, file_path, start_frame=29, frame_rate=30):
        file_name = file_path.split("/")[-1]
        set_num, v_num, frame_num = file_name[:-4].split("_")
        if int(frame_num) >= start_frame and int(frame_num) % frame_rate == 29:
            target_frames.setdefault(set_num, {}).setdefault(v_num, []).append(file_path)
            return 1
        else:
            return 0

    def get_target_frames(image_set_list, image_path):
        target_frames = {}
        total_frames = 0
        for set_num in image_set_list:
            file_pattern = "{}/set{}_V*".format(image_path, set_num)
            print(file_pattern)
            file_list = sorted(glob.glob(file_pattern), key=natural_keys)
            for file_path in file_list:
                total_frames += insert_frame(target_frames, file_path)
        return target_frames, total_frames

    def detection_to_file(target_path, v_num, file_list, total_frames, current_frames, thresh=0):
        timer = Timer()
        w = open("{}/{}.txt".format(target_path, v_num), "w")
        for file_index, file_path in enumerate(file_list):
            file_name = file_path.split("/")[-1]
            set_num, v_num, frame_num = file_name[:-4].split("_")
            frame_num = str(int(frame_num) + 1)

            timer.tic()
            dets = im_detect(net, file_path)
            timer.toc()

            # get pedestrian dets
            print('Detection Time:{:.3f}s on {}  {}/{} images'.format(timer.average_time,
                                                                      file_name,
                                                                      current_frames + file_index + 1,
                                                                      total_frames))

            inds = np.where(dets[:, -1] >= thresh)[0]
            for i in inds:
                bbox = dets[i, :4]
                score = dets[i, -1]

                # Fix bug 6
                x = bbox[0]
                y = bbox[1]
                width = bbox[2] - x
                length = bbox[3] - y
                if score * 100 > 70:
                    print("{},{},{},{},{},{}\n".format(frame_num, x, y, width, length, score * 100))

                w.write("{},{},{},{},{},{}\n".format(frame_num, x, y, width, length, score * 100))

        w.close()
        print("Evalutaion file {} has been writen".format(w.name))
        return file_index + 1

    output_path = os.path.join(OUTPUT_PATH, DETECTION_NAME)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    print('result output dir', output_path)

    image_set_list = ["06", "07", "08", "09", "10"]
    target_frames, total_frames = get_target_frames(image_set_list, IMG_PATH)

    current_frames = 0
    for set_num in target_frames:
        target_path = os.path.join(output_path, set_num)
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        for v_num, file_list in target_frames[set_num].items():
            current_frames += detection_to_file(target_path, v_num, file_list, total_frames, current_frames)


def prepare_model(args):
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

    return fasterRCNN


if __name__ == "__main__":
    args = parse_args()
    global DETECTION_NAME
    DETECTION_NAME = args.dt_name
    model = prepare_model(args)
    write_caltech_results_file(model)