from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tkinter as tk
from tkinter import filedialog

from PIL import Image, ImageTk

import argparse

import csv
import os
import shutil
import time
import sys

sys.path.append("../lib")

import cv2
import numpy as np
import random
from PIL import Image

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision

import _init_paths
import models
import tqdm

from config import cfg
from config import update_config
from core.inference import get_multi_stage_outputs
from core.inference import aggregate_results
from core.nms import pose_nms
from core.match import match_pose_to_heatmap
from utils.transforms import resize_align_multi_scale
from utils.transforms import get_final_preds
from utils.transforms import get_multi_scale_size
from utils.transforms import up_interpolate

CTX = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

COCO_KEYPOINT_INDEXES = {
    0: 'nose',
    1: 'left_eye',
    2: 'right_eye',
    3: 'left_ear',
    4: 'right_ear',
    5: 'left_shoulder',
    6: 'right_shoulder',
    7: 'left_elbow',
    8: 'right_elbow',
    9: 'left_wrist',
    10: 'right_wrist',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle'
}

CROWDPOSE_KEYPOINT_INDEXES = {
    0: 'left_shoulder',
    1: 'right_shoulder',
    2: 'left_elbow',
    3: 'right_elbow',
    4: 'left_wrist',
    5: 'right_wrist',
    6: 'left_hip',
    7: 'right_hip',
    8: 'left_knee',
    9: 'right_knee',
    10: 'left_ankle',
    11: 'right_ankle',
    12: 'head',
    13: 'neck'
}

def get_pose_estimation_prediction(cfg, model, image, vis_thre, transforms):
    # size at scale 1.0
    base_size, center, scale = get_multi_scale_size(
        image, cfg.DATASET.INPUT_SIZE, 1.0, 1.0
    )

    with torch.no_grad():
        heatmap_sum = 0
        poses = []

        for scale in sorted(cfg.TEST.SCALE_FACTOR, reverse=True):
            image_resized, center, scale_resized = resize_align_multi_scale(
                image, cfg.DATASET.INPUT_SIZE, scale, 1.0
            )

            image_resized = transforms(image_resized)
            image_resized = image_resized.unsqueeze(0).cuda()

            heatmap, posemap = get_multi_stage_outputs(
                cfg, model, image_resized, cfg.TEST.FLIP_TEST
            )
            heatmap_sum, poses = aggregate_results(
                cfg, heatmap_sum, poses, heatmap, posemap, scale
            )

        heatmap_avg = heatmap_sum / len(cfg.TEST.SCALE_FACTOR)
        poses, scores = pose_nms(cfg, heatmap_avg, poses)

        if len(scores) == 0:
            return []
        else:
            if cfg.TEST.MATCH_HMP:
                poses = match_pose_to_heatmap(cfg, poses, heatmap_avg)

            final_poses = get_final_preds(
                poses, center, scale_resized, base_size
            )

        final_results = []
        for i in range(len(scores)):
            if scores[i] > vis_thre:
                final_results.append(final_poses[i])

        if len(final_results) == 0:
            return []

    return final_results

def show_skeleton(img, kpts, color=(0,255,0), thr=0.5):
    kpts = np.array(kpts).reshape(-1, 3)
    skelenton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10],
                 [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

    points_num = [num for num in range(17)]
    for sk in skelenton:

        pos1 = (int(kpts[sk[0] - 1, 0]), int(kpts[sk[0] - 1, 1]))
        pos2 = (int(kpts[sk[1] - 1, 0]), int(kpts[sk[1] - 1, 1]))
        if pos1[0] > 0 and pos1[1] > 0 and pos2[0] > 0 and pos2[1] > 0 and kpts[sk[0] - 1, 2] > thr and kpts[
            sk[1] - 1, 0] > thr:
            cv2.line(img, pos1, pos2, color, 2, 8)
    for points in points_num:
        pos = (int(kpts[points, 0]), int(kpts[points, 1]))
        if pos[0] > 0 and pos[1] > 0 and kpts[points, 2] > thr:
            cv2.circle(img, pos, 1, (0, 0, 255), -1)  # 为肢体点画红色实心圆
    return img


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg', type=str, default='/home/omnisky/project/DEKR-main/experiments/coco/inference_demo_coco.yaml')
    parser.add_argument('--imagesDir', type=str, default='/home/omnisky/project/DEKR-main/image')
    parser.add_argument('--outputDir', type=str, default='/home/omnisky/project/DEKR-main/pose')
    parser.add_argument('--inferenceFps', type=int, default=10)
    parser.add_argument('--visthre', type=float, default=0)
    parser.add_argument('opts',
                        help='Modify config options using the command-line',
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    # args expected by supporting codebase
    args.modelDir = ''
    args.logDir = ''
    args.dataDir = ''
    args.prevModelDir = ''
    return args




def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg', type=str, default='/home/omnisky/project/DEKR-main/experiments/coco/inference_demo_coco.yaml')
    parser.add_argument('--imagesDir', type=str, default='/home/omnisky/project/DEKR-main/image')
    parser.add_argument('--outputDir', type=str, default='/home/omnisky/project/DEKR-main/pose')
    parser.add_argument('--inferenceFps', type=int, default=10)
    parser.add_argument('--visthre', type=float, default=0)
    parser.add_argument('opts',
                        help='Modify config options using the command-line',
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    # args expected by supporting codebase
    args.modelDir = ''
    args.logDir = ''
    args.dataDir = ''
    args.prevModelDir = ''
    return args


class KeypointDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("关键点检测系统")
        # self.root.iconbitmap('/home/omnisky/project/DEKR-main/tools/101669.ico')
        self.image_label = tk.Label(self.root)
        self.image_label.pack(padx=10, pady=10)

        self.load_button = tk.Button(self.root, text="加载图片", command=self.load_image, bg='orange', height=2, width=8,font=('Helvetica', 15))
        self.load_button.pack(pady=5)

        self.detect_button = tk.Button(self.root, text="检测关键点", command=self.detect_keypoints,bg='green',height=2, width=8,font=('Helvetica', 15))
        self.detect_button.pack(pady=5)



        self.image = None

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image = cv2.imread(file_path)
            self.display_image(self.image)
，
    def detect_keypoints(self):

        if self.image is not None:


            # cudnnelated setting
            cudnn.benchmark = cfg.CUDNN.BENCHMARK
            torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
            torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

            args = parse_args()
            update_config(cfg, args)
            # pose_dir = prepare_output_dirs(args.outputDir)
            # csv_output_rows = []

            pose_model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(
                cfg, is_train=False
            )
            if cfg.TEST.MODEL_FILE:
                print('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
                pose_model.load_state_dict(torch.load(
                    cfg.TEST.MODEL_FILE), strict=False)
            else:
                raise ValueError('expected model defined in config at TEST.MODEL_FILE')

            pose_model.to(CTX)
            pose_model.eval()
            count = 0
            # image_bgr = cv2.imread(file_path)
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            # self.display_image(gray_image)
            image_pose = gray_image.copy()
            image_debug = image_pose.copy()

            pose_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
            h,w, _ = image_pose.shape
            pose_preds = get_pose_estimation_prediction(cfg, pose_model, image_pose, args.visthre, transforms=pose_transform)



            skeleton_color = [(154, 194, 182), (123, 151, 138), (0, 208, 244), (8, 131, 229), (18, 87, 220)]
            new_csv_row = []
            for coords in pose_preds:
                # 可视化部分
                # color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
                show_skeleton(image_debug, coords)
            # detector = cv2.ORB_create()
            # keypoints, _ = detector.detectAndCompute(gray_image, None)
            # image_with_keypoints = cv2.drawKeypoints(self.image, keypoints, None)
            self.display_image(image_debug)
        else:
            tk.messagebox.showwarning("Warning", "Please load an image first.")

    def display_image(self, image):
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        self.image_label.configure(image=image)
        self.image_label.image = image


if __name__ == "__main__":
    root = tk.Tk()
    app = KeypointDetectionApp(root)

    root.mainloop()
