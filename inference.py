# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.
"""

import argparse
import os
import sys
from pathlib import Path
import glob

import cv2
import torch

OUTPUT_NAME = "data.csv"

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import LoadImages
from utils.general import LOGGER, check_img_size, check_requirements, non_max_suppression, print_args, scale_coords
from utils.torch_utils import select_device, time_sync

import torch
import torch.nn as nn
from torchvision.models import resnet18, vgg11

import numpy as np

from script.Dataset import generate_bins, DetectedObject
from library.Math import *
from library.Plotting import *
from script import Model, ClassAverages
from script.Model import ResNet, ResNet18, VGG11
import tqdm

# model factory to choose model
model_factory = {
    'resnet': resnet18(pretrained=True),
    'resnet18': resnet18(pretrained=True),
    # 'vgg11': vgg11(pretrained=True)
}
regressor_factory = {
    'resnet': ResNet,
    'resnet18': ResNet18,
    'vgg11': VGG11
}

class Bbox:
    def __init__(self, box_2d, class_, line, id):
        self.box_2d = box_2d
        self.detected_class = class_
        self.line = line
        self.id = id

def detect3d(
    reg_weights,
    model_select,
    source,
    show_result,
    save_result,
    output_path
    ):

    # Directory

    with open(OUTPUT_NAME, "w") as fp:
        fp.write("id,line,center_x,center_y,center_z,length,width,height,theta\n")

    imgs_path = []
    labels_path = []
    calibs_path = []
    with open(os.path.join(source, "test.txt"), 'r') as f:
        lines = f.readlines()
        for line in tqdm.tqdm(lines, desc="Loading Images"):
            file_id = line.strip().split(".")[0]
            img_path = os.path.join(source, "image_2", file_id + ".png")
            label_path = os.path.join(source, "label_2", file_id + ".txt")
            calib_path = os.path.join(source, "calib", file_id + ".txt")

            # Assert that the files exist
            assert os.path.isfile(img_path)
            assert os.path.isfile(label_path)
            assert os.path.isfile(calib_path)

            imgs_path.append(img_path)
            labels_path.append(label_path)
            calibs_path.append(calib_path)

    # load model
    base_model = model_factory[model_select]
    regressor = regressor_factory[model_select](model=base_model).cuda()

    # load weight
    checkpoint = torch.load(reg_weights)
    regressor.load_state_dict(checkpoint['model_state_dict'])
    regressor.eval()

    averages = ClassAverages.ClassAverages()
    angle_bins = generate_bins(2)

    # loop images

    items = list(enumerate(zip(imgs_path, labels_path, calibs_path)))
    titems = tqdm.tqdm(items, desc="Detecting 3D")

    for i, (img_path, label_path, calib_path) in titems:
        # read image
        img = cv2.imread(img_path)
        
        # Run detection 2d
        dets = detect2d(label=label_path)

        for det in dets:
            if not averages.recognized_class(det.detected_class):
                continue
            try: 
                detectedObject = DetectedObject(img, det.detected_class, det.box_2d, calib_path)
            except Exception as e:
                print(e)
                raise e

            theta_ray = detectedObject.theta_ray
            input_img = detectedObject.img
            proj_matrix = detectedObject.proj_matrix
            box_2d = det.box_2d
            detected_class = det.detected_class

            input_tensor = torch.zeros([1,3,224,224]).cuda()
            input_tensor[0,:,:,:] = input_img

            # predict orient, conf, and dim
            [orient, conf, dim] = regressor(input_tensor)
            orient = orient.cpu().data.numpy()[0, :, :]
            conf = conf.cpu().data.numpy()[0, :]
            dim = dim.cpu().data.numpy()[0, :]

            dim += averages.get_item(detected_class)

            argmax = np.argmax(conf)
            orient = orient[argmax, :]
            cos = orient[0]
            sin = orient[1]
            alpha = np.arctan2(sin, cos)
            alpha += angle_bins[argmax]
            alpha -= np.pi

            # plot 3d detection
            with open(OUTPUT_NAME, "a") as fp:
                fp.write(f"{det.id},{det.line},")

            plot3d(img, proj_matrix, box_2d, dim, alpha, theta_ray)
            i += 1

        if show_result:
            cv2.imshow('3d detection', img)
            cv2.waitKey(0)

        if save_result and output_path is not None:
            try:
                os.mkdir(output_path)
            except:
                pass
            cv2.imwrite(f'{output_path}/{i:03d}.png', img)

@torch.no_grad()
def detect2d(
    label,
    ):


    bbox_list = []
    seen = 0

    fileid = os.path.basename(label).split(".")[0]
    with open(label, "r") as fp:
        for i, line in enumerate(fp):
            line = line.strip().split()
            if line[0] == "Car" and int(line[-2]) != 0:
                top_left = (int(round(float(line[4]))), int(round(float(line[5]))))
                bottom_right = (int(round(float(line[6]))), int(round(float(line[7]))))
                box = [top_left, bottom_right]
                bbox_list.append(Bbox(box, line[0].lower(), i, fileid))
                seen += 1

    # Print results
    # top_left, bottom_right = (xyxy_[0], xyxy_[1]), (xyxy_[2], xyxy_[3])
    # bbox = [top_left, bottom_right]
    # bbox_list.append(Bbox(bbox, label))

    return bbox_list

def plot3d(
    img,
    proj_matrix,
    box_2d,
    dimensions,
    alpha,
    theta_ray,
    img_2d=None
    ):

    # the math! returns X, the corners used for constraint
    location, X = calc_location(dimensions, proj_matrix, box_2d, alpha, theta_ray)

    orient = alpha + theta_ray

    if img_2d is not None:
        plot_2d_box(img_2d, box_2d)

    with open(OUTPUT_NAME, "a") as fp:
        fp.write(f"{location[0]},{location[1]},{location[2]},{dimensions[0]},{dimensions[1]},{dimensions[2]},{orient}\n")

    plot_3d_box(img, proj_matrix, orient, dimensions, location) # 3d boxes

    return location

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'eval/image_2', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', default=[0, 2, 3, 5], nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--reg_weights', type=str, default='weights/epoch_10.pkl', help='Regressor model weights')
    parser.add_argument('--model_select', type=str, default='resnet', help='Regressor model list: resnet, vgg, eff')
    parser.add_argument('--show_result', action='store_true', help='Show Results with imshow')
    parser.add_argument('--save_result', action='store_true', help='Save result')
    parser.add_argument('--output_path', type=str, default=ROOT / 'output', help='Save output pat')

    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt

def main(opt):
    detect3d(
        reg_weights=opt.reg_weights,
        model_select=opt.model_select,
        source=opt.source,
        show_result=opt.show_result,
        save_result=opt.save_result,
        output_path=opt.output_path
    )

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)