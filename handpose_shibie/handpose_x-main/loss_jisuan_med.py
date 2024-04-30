import os
import mediapipe as mp
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import  sys

from utils.model_utils import *
from utils.common_utils import *
from hand_data_iter.datasets import *

from multiprocessing import freeze_support
from models.resnet import resnet18,resnet34,resnet50,resnet101
from models.squeezenet import squeezenet1_1,squeezenet1_0
from models.shufflenetv2 import ShuffleNetV2
from models.shufflenet import ShuffleNet
from models.mobilenetv2 import MobileNetV2
from models.rexnetv1 import ReXNetV1

from torchvision.models import shufflenet_v2_x1_5 ,shufflenet_v2_x1_0 , shufflenet_v2_x2_0

from loss.loss import *
import cv2
import time
import json
from datetime import datetime


def de_pts(num):
    # 定义文件夹路径
    folder_path = r'D:\hand_data\handpose_datasets_v1-2021-01-31\handpose_datasets_v1'
    idx = 0
    data_pt_x = []
    data_pt_y = []
    data_img_width = []
    data_img_height = []
    pts_t_z = []
    imgs = []
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 检查文件名是否以 '.json' 结尾
        if filename.endswith('.json'):
            # 构建 JSON 文件的完整路径
            json_file_path = os.path.join(folder_path, filename)

            # 读取 JSON 文件
            with open(json_file_path, 'r') as f:
                data = json.load(f)

            # 打印 JSON 数据
            # print(data)
            # 获取 info 字段
            info = data['info']

            # 提取第一个 info 字段中的 pts 字段
            pts = info[0]['pts']

            # 创建空列表来存储提取出的 x 和 y 坐标
            x_coordinates = []
            y_coordinates = []

            # 遍历 pts 中的所有键值对
            for key, value in pts.items():
                # 提取 x 和 y 坐标，并将其转换为整数
                x = int(value['x'])
                y = int(value['y'])

                # 将坐标添加到对应的列表中
                x_coordinates.append(x)
                y_coordinates.append(y)

            data_pt_x.append(x_coordinates)
            data_pt_y.append(y_coordinates)

            # 打印提取出的 x 和 y 坐标
            # print("X 坐标:", x_coordinates)
            # print("Y 坐标:", y_coordinates)



        # 检查文件名是否以 '.jpg' 结尾
        elif filename.endswith('.jpg'):
            # 构建图像文件的完整路径
            image_file_path = os.path.join(folder_path, filename)

            # 读取图像文件
            image = cv2.imread(image_file_path)
            imgs.append(image)
            height, width, channels = image.shape

            data_img_width.append(width)
            data_img_height.append(height)
        idx += 1

        if idx == num*2:
            break
    # print(idx)
    for i in range(0, num):
        width = data_img_width[i]
        height = data_img_height[i]
        pts_t = []
        for j in range(len(data_pt_x[i])):
            x_gui = data_pt_x[i][j] / width
            y_gui = data_pt_y[i][j] / height
            # print(x_gui)
            pts_t.append(x_gui)
            pts_t.append(y_gui)
        pts_t_z.append(pts_t)

    # print(pts_t_z)
    pts_t_z_tensor = torch.tensor(pts_t_z)
    pts_t_z_tensor = pts_t_z_tensor.to(torch.float64)
    # print(pts_t_z_tensor)

    return pts_t_z_tensor

def mediapipe_img_out(num):
    ############# 基础设置 ####################
    model_path = 'D:/hand_data/Madiapipe/Model/hand_landmarker.task'
    folder_path = r'D:\hand_data\handpose_datasets_v1-2021-01-31\handpose_datasets_v1'
    idx = 0
    data_pt_x = []
    data_pt_y = []
    pts_t_z = []
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    # Create a hand landmarker instance with the image mode:
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE)
    with HandLandmarker.create_from_options(options) as landmarker:
        for filename in os.listdir(folder_path):
            if filename.endswith('.jpg'):
                image_file_path = os.path.join(folder_path, filename)
                mp_image = mp.Image.create_from_file(image_file_path)
                hand_landmarker_result = landmarker.detect(mp_image)
                # print(hand_landmarker_result)
                hand_mark = hand_landmarker_result.hand_landmarks
                # 初始化存储x和y坐标的列表
                handpose_x = []
                handpose_y = []

                # 遍历每个关键点
                for j in range(0, 21):
                    if hand_mark != []:
                        handpose_x.append(hand_mark[0][j].x)
                        handpose_y.append(hand_mark[0][j].y)

                # 打印结果（可选）
                # print("X Coordinates:", handpose_x)
                # print("Y Coordinates:", handpose_y)
                data_pt_x.append(handpose_x)
                data_pt_y.append(handpose_y)


            idx += 1
            if idx == num * 2:
                break

    for i in range(0, num):
        pts_t = []
        for j in range(len(data_pt_x[i])):
            pts_t.append(data_pt_x[i][j])
            pts_t.append(data_pt_y[i][j])
        pts_t_z.append(pts_t)

    # print(pts_t_z)
    pts_t_z_tensor = torch.tensor(pts_t_z)
    pts_t_z_tensor = pts_t_z_tensor.to(torch.float64)
    # print(pts_t_z_tensor)

    return pts_t_z_tensor



if __name__ == '__main__':
    output = mediapipe_img_out(5)
    pts_ = de_pts(5)
    loss = got_total_wing_loss(output, pts_.float())
    print(loss)



