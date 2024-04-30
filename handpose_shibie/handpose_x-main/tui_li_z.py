#-*-coding:utf-8-*-
# date:2021-04-5
# Author: Eric.Lee
# function: Inference

import os
import argparse
import torch
import sys
import random
import torch.nn as nn
import numpy as np

import time
import datetime
import os
import cv2
import math
from datetime import datetime
import torch.nn.functional as F
import matplotlib.pyplot as plt
import red_shibie
import chaifen_shipin
import test_hand_video
import mediapipe_video

from models.resnet import resnet18,resnet34,resnet50,resnet101
from models.squeezenet import squeezenet1_1,squeezenet1_0
from models.shufflenetv2 import ShuffleNetV2
from models.shufflenet import ShuffleNet
from models.mobilenetv2 import MobileNetV2
from torchvision.models import shufflenet_v2_x1_5 ,shufflenet_v2_x1_0 , shufflenet_v2_x2_0
from models.rexnetv1 import ReXNetV1

from utils.common_utils import *
import copy
from hand_data_iter.datasets import draw_bd_handpose

def clear_folder(folder_path):
    # 遍历文件夹中的所有文件和子文件夹
    for root, dirs, files in os.walk(folder_path):
        # 删除文件
        for file in files:
            file_path = os.path.join(root, file)
            os.remove(file_path)
        # 删除子文件夹
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            os.rmdir(dir_path)

def angel_su_jisuan(angle_radians, time):
    angle_su = []
    angle_su.append(0)

    for i in range(len(angle_radians)-1):
        angle_su_i = (angle_radians[i+1]-angle_radians[i])/(time[i+1]-time[i])
        angle_su.append(angle_su_i)

    return angle_su

def angle_sujia_jisuan(angle_su, time):
    angle_sujia = []
    angle_sujia.append(0)

    for i in range(len(angle_su)-1):
        angle_sujia_i = (angle_su[i+1]-angle_su[i])/(time[i+1]-time[i])
        angle_sujia.append(angle_sujia_i)

    return angle_sujia





def angel_jisuan(flag_ang_i, flag_ang_j,  hand_joint_x, hand_joint_y, idx):
    angle = []
    angle_radian = []
    if flag_ang_i ==  5 or 9 or 13 or 17:
        fir = 0
    elif flag_ang_i == 2:
        fir = 1
    else:
        fir = flag_ang_i - 1
    for i in range(0, idx):
        a = np.array([hand_joint_x[i][fir], hand_joint_y[i][fir]])
        b = np.array([hand_joint_x[i][flag_ang_i], hand_joint_y[i][flag_ang_i]])
        c = np.array([hand_joint_x[i][flag_ang_j], hand_joint_y[i][flag_ang_j]])
        ab = b - a
        bc = c - b
        cross_product = np.cross(bc, ab)
        dian_ji = np.dot(ab, bc)
        ab_m = np.linalg.norm(ab)
        bc_m = np.linalg.norm(bc)
        cos_angle = dian_ji/(ab_m * bc_m)
        if cross_product < 0:
            angle_radians = -np.arccos(np.clip(cos_angle, -1.0, 1.0))
        else:
            angle_radians = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        angle_degrees = np.degrees(angle_radians)
        # print("第%d个关节和第%d个关节的夹角为：" % (flag_ang_i, flag_ang_j))
        # print("%d度" % angle_degrees)
        angle_radian.append(angle_radians)
        angle.append(angle_degrees)

    img_x = [i for i in range(1, idx + 1)]
    img_x = [i * 0.033333333 for i in img_x]

    # plt.rcParams['font.family'] = ['SimHei']  # 设置中文字体为黑体，可以根据需要更改为其他字体
    plt.rcParams['font.family'] = ['Microsoft YaHei']  # 微软雅黑
    plt.figure(flag_ang_i)
    plt.figure(figsize=(30, 15))
    plt.rcParams.update({'font.size': 20})
    plt.plot(img_x, angle)
    plt.title('第%d个和第%d个关键点形成关节角的角度变化图' % (flag_ang_i, flag_ang_j))
    plt.xlabel('time/s')
    plt.ylabel('angle/°')
    plt.savefig('./picture/hand_ang_{}_{}.png'.format(flag_ang_i, flag_ang_j))
    # plt.show()

    angle_su = angel_su_jisuan(angle_radian, img_x)
    angle_sujia = angle_sujia_jisuan(angle_su, img_x)

    figure_num = flag_ang_i+100
    plt.rcParams['font.family'] = ['Microsoft YaHei']  # 微软雅黑
    plt.figure(figure_num)
    plt.figure(figsize=(30, 15))
    plt.rcParams.update({'font.size': 20})
    plt.plot(img_x, angle_su)
    plt.title('第%d个和第%d个关键点形成关节角的角速度变化图' % (flag_ang_i, flag_ang_j))
    plt.xlabel('time/(s)')
    plt.ylabel('angle_xu/(rad/s)')
    plt.savefig('./picture/hand_ang_su_{}_{}.png'.format(flag_ang_i, flag_ang_j))

    figure_num = flag_ang_i+200
    plt.rcParams['font.family'] = ['Microsoft YaHei']  # 微软雅黑
    plt.figure(figure_num)
    plt.figure(figsize=(30, 15))
    plt.rcParams.update({'font.size': 20})
    plt.plot(img_x, angle_sujia)
    plt.title('第%d个和第%d个关键点形成关节角的角加速度变化图' % (flag_ang_i, flag_ang_j))
    plt.xlabel('time/(s)')
    plt.ylabel('angle_xu/(rad/s^2)')
    plt.savefig('./picture/hand_ang_su_jia_{}_{}.png'.format(flag_ang_i, flag_ang_j))

    if flag_ang_i == 5 or 9 or 13 or 17:
        min_ang = -20
        max_ang = 90
    elif flag_ang_i == 6 or 10 or 14 or 18:
        min_ang = -30
        max_ang = 100
    elif flag_ang_i == 7 or 11 or 15 or 19:
        min_ang = -30
        max_ang = 70
    elif flag_ang_i == 2:
        min_ang = -40
        max_ang = 60
    else:
        min_ang = -30
        max_ang = 80

    max_value = np.max(np.clip(angle, min_ang, max_ang))
    min_value = np.min(np.clip(angle, min_ang, max_ang))

    L = max_ang - min_ang
    fai = max_value - min_value

    socre = (L - fai)/L*0.75
    print('第%d个和第%d个关键点形成关节角得分：' % (flag_ang_i, flag_ang_j))
    print(socre)

    plt.close()
    return socre


def hongse(idx, hand_joint_x, hand_joint_y, red, blue, frame_time, num):
    ############# 判断是否到达红色标记坐标 ###################
    radius = 15
    neighbor_pixels = [[] for _ in range(idx)]
    # 遍历以当前像素点为中心的周围区域
    for p in range(0, idx):
        for i in range(int(hand_joint_x[p][num] - radius), int(hand_joint_x[p][num] + radius + 1)):
            for j in range(int(hand_joint_y[p][num] - radius), int(hand_joint_y[p][num] + radius + 1)):
                # 计算当前像素点到中心像素点的距离
                distance = ((i - hand_joint_x[p][num]) ** 2 + (j - hand_joint_y[p][num]) ** 2) ** 0.5
                random_number = random.random()
                # 如果距离小于等于半径，则将该像素点的坐标添加到列表中
                if distance <= radius and random_number <= 0.5:
                    neighbor_pixels[p].append((i, j))
    # print(neighbor_pixels)

    flag_T = 0
    # flag_F = 0
    ten_or_more = []
    for p in range(0, idx):
        for i in range(len(neighbor_pixels[p])):
            if neighbor_pixels[p][i] in red[p]:
                # if (117, 84) in red[i]:
                flag_T = 1
            # else:
            #     flag_F = 0
            if neighbor_pixels[p][i] in blue[p]:
                flag_T = 2
        if flag_T == 1:
            print("第%d张图片到达红色按钮位置：" % (p + 1))
            ten_or_more.append(flag_T)
        elif flag_T == 2:
            print("第%d张图片到达蓝色按钮位置：" % (p + 1))
            ten_or_more.append(flag_T)
        else:
            print("第%d张图片未达到按钮位置：" % (p + 1))
            ten_or_more.append(flag_T)

        flag_T = 0

    xiaohao_time = 0
    for i in range(len(ten_or_more)-4):
        if ten_or_more[i] and ten_or_more[i + 1] and ten_or_more[i + 2] and ten_or_more[i + 3] and ten_or_more[i + 4] == 1:
            xiaohao_time = frame_time * (i + 1)
            print("抵达按钮消耗时间: %f s" % xiaohao_time)
            break
    if xiaohao_time == 0:
        print("未抵达红色按钮")
        xiaohao_time = 100

    # flag_T = 0
    # # flag_F = 0
    # ten_or_more = []
    # for p in range(0, idx):
    #     for i in range(len(neighbor_pixels[p])):
    #         if neighbor_pixels[p][i] in blue[p]:
    #             # if (117, 84) in red[i]:
    #             flag_T = 1
    #         # else:
    #         #     flag_F = 0
    #     if flag_T == 1:
    #         print("第%d张图片是否达到蓝色按钮位置：" % (p + 1))
    #         print('Ture')
    #         ten_or_more.append(flag_T)
    #     else:
    #         print("第%d张图片是否达到蓝色按钮位置：" % (p + 1))
    #         print('False')
    #         ten_or_more.append(flag_T)
    #     flag_T = 0

    xiaohao_time_blue = 0
    for i in range(len(ten_or_more)-4):
        if ten_or_more[i] and ten_or_more[i + 1] and ten_or_more[i + 2] and ten_or_more[i + 3] and ten_or_more[i + 4] == 2:
            xiaohao_time_blue = frame_time * (i + 1)
            print("抵达按钮消耗时间: %f s" % xiaohao_time_blue)
            break
    if xiaohao_time_blue == 0:
        print("未抵达蓝色按钮")
        xiaohao_time_blue = 200


    return xiaohao_time, xiaohao_time_blue



def jisuan_SIE(socre_2_3, socre_3_4, socre_5_6, socre_6_7, socre_7_8, socre_9_10, socre_10_11, socre_11_12, socre_13_14, socre_14_15, socre_15_16,
                               socre_17_18, socre_18_19, socre_19_20):
    socre = 1-(socre_2_3*0.27 + socre_3_4*0.13 + socre_5_6*0.09 + socre_6_7*0.07 + socre_7_8*0.04 + socre_9_10*0.09 + socre_10_11*0.07 + socre_11_12*0.04 + socre_13_14*0.045 + socre_14_15*0.035 + socre_15_16*0.02 + socre_17_18*0.045 + socre_18_19*0.035 + socre_19_20*0.02)
    socre = socre*100
    print("SIE得分：%d%%" % socre)
    return socre

def touchs1(time_red, time_blue):
    t_z_red = 1.1
    t_z_blue = 2.5
    touchs = (t_z_red/time_red)*0.5+(t_z_blue/time_blue)*0.5
    if touchs > 1:
        touchs = 1

    return touchs

def touchs(time_8_red, time_8_blue, time_4_red, time_4_blue, time_12_red, time_12_blue, time_16_red, time_16_blue, time_20_red, time_20_blue):
    socre_8 = touchs1(time_8_red, time_8_blue)
    socre_4 = touchs1(time_4_red, time_4_blue)
    socre_12 = touchs1(time_12_red, time_12_blue)
    socre_16 = touchs1(time_16_red, time_16_blue)
    socre_20 = touchs1(time_20_red, time_20_blue)

    touchs = socre_4*0.3+socre_8*0.4+socre_12*0.2+socre_16*0.05+socre_20*0.05

    return touchs*100






def extract_number(filename):
    # 从文件名中提取数字部分
    return int(''.join(filter(str.isdigit, filename)))


if __name__ == "__main__":



    parser = argparse.ArgumentParser(description=' Project Hand Pose Inference')
    # parser.add_argument('--model_path', type=str, default = './weights/ReXNetV1-size-256-wingloss102-0.122.pth',
    #     help = 'model_path') # 模型路径
    parser.add_argument('--model_path', type=str, default = 'D:/hand_data/Madiapipe/Model/hand_landmarker.task',
        help = 'model_path') # 模型路径
    parser.add_argument('--test_video_path', type=str,
                        default='D:/hand_data/test_video/test_shizhi_1.mp4',
                        help='test_video_path')  # 测试视频路径——食指
    parser.add_argument('--test_video_path_sizhi', type=str,
                        default='D:/hand_data/test_video/test_sizhi.mp4',
                        help='test_video_path')  # 测试视频路径——四指角度
    parser.add_argument('--test_video_path_muzhi', type=str,
                        default='D:/hand_data/test_video/test_muzhi.mp4',
                        help='test_video_path')  # 测试视频路径——拇指角度
    parser.add_argument('--test_video_path_muzhi_1', type=str,
                        default='D:/hand_data/test_video/test_muzhi_1.mp4',
                        help='test_video_path')  # 测试视频路径——拇指
    parser.add_argument('--test_video_path_zhongzhi_1', type=str,
                        default='D:/hand_data/test_video/test_zhongzhi_1.mp4',
                        help='test_video_path')  # 测试视频路径——中指
    parser.add_argument('--test_video_path_huanzhi_1', type=str,
                        default='D:/hand_data/test_video/test_huanzhi_1.mp4',
                        help='test_video_path')  # 测试视频路径——环指
    parser.add_argument('--test_video_path_xiaozhi_1', type=str,
                        default='D:/hand_data/test_video/test_xiaozhi_1.mp4',
                        help='test_video_path')  # 测试视频路径——小指
    parser.add_argument('--model', type=str, default = "mediapipe",
        help = '''model : resnet_34,resnet_50,resnet_101,squeezenet1_0,squeezenet1_1,shufflenetv2,shufflenet,mobilenetv2
            shufflenet_v2_x1_5 ,shufflenet_v2_x1_0 , shufflenet_v2_x2_0,ReXNetV1,mediapipe''') # 模型类型
    parser.add_argument('--num_classes', type=int , default = 42,
        help = 'num_classes') #  手部21关键点， (x,y)*2 = 42
    parser.add_argument('--GPUS', type=str, default = '0',
        help = 'GPUS') # GPU选择
    parser.add_argument('--test_path_shi', type=str, default = './image_shi/',
        help = 'test_path') # 测试图片路径
    parser.add_argument('--test_path_mu', type=str, default='./image_mu/',
                        help='test_path')  # 测试图片路径
    parser.add_argument('--test_path_zhong', type=str, default='./image_zhong/',
                        help='test_path')  # 测试图片路径
    parser.add_argument('--test_path_huan', type=str, default='./image_huan/',
                        help='test_path')  # 测试图片路径
    parser.add_argument('--test_path_xiao', type=str, default='./image_xiao/',
                        help='test_path')  # 测试图片路径
    parser.add_argument('--img_size', type=tuple , default = (256,256),
        help = 'img_size') # 输入模型图片尺寸
    parser.add_argument('--vis', type=bool , default = True,
        help = 'vis') # 是否可视化图片

    print('\n/******************* {} ******************/\n'.format(parser.description))
    #--------------------------------------------------------------------------
    ops = parser.parse_args()# 解析添加参数
    #--------------------------------------------------------------------------
    print('----------------------------------')

    unparsed = vars(ops) # parse_args()方法的返回值为namespace，用vars()内建函数化为字典
    for key in unparsed.keys():
        print('{} : {}'.format(key,unparsed[key]))

    #---------------------------------------------------------------------------
    os.environ['CUDA_VISIBLE_DEVICES'] = ops.GPUS

    test_path_shi =  ops.test_path_shi # 测试图片文件夹路径
    test_path_mu = ops.test_path_mu
    test_path_zhong = ops.test_path_zhong
    test_path_huan = ops.test_path_huan
    test_path_xiao = ops.test_path_xiao

    test_video_path = ops.test_video_path # 测试视频路径
    test_video_path_sizhi = ops.test_video_path_sizhi
    test_video_path_muzhi = ops.test_video_path_muzhi
    test_video_path_muzhi_1 = ops.test_video_path_muzhi_1
    test_video_path_zhongzhi_1 = ops.test_video_path_zhongzhi_1
    test_video_path_huanzhi_1 = ops.test_video_path_huanzhi_1
    test_video_path_xiaozhi_1 = ops.test_video_path_xiaozhi_1

    clear_folder(test_path_shi) # 清空文件夹
    clear_folder(test_path_mu)
    clear_folder(test_path_zhong)
    clear_folder(test_path_huan)
    clear_folder(test_path_xiao)
    ###################### 将视频文件转为一秒三十帧的图片们 ###########################
    video_path = test_video_path
    output_dir = test_path_shi
    frame_time = chaifen_shipin.split_video_to_frames(video_path, output_dir)

    video_path = test_video_path_muzhi_1
    output_dir = test_path_mu
    chaifen_shipin.split_video_to_frames(video_path, output_dir)

    video_path = test_video_path_zhongzhi_1
    output_dir = test_path_zhong
    chaifen_shipin.split_video_to_frames(video_path, output_dir)

    video_path = test_video_path_huanzhi_1
    output_dir = test_path_huan
    chaifen_shipin.split_video_to_frames(video_path, output_dir)

    video_path = test_video_path_xiaozhi_1
    output_dir = test_path_xiao
    chaifen_shipin.split_video_to_frames(video_path, output_dir)

    #---------------------------------------------------------------- 构建模型
    print('use model : %s'%(ops.model))

    if ops.model == 'resnet_50':
        model_ = resnet50(num_classes = ops.num_classes,img_size=ops.img_size[0])
    elif ops.model == 'resnet_18':
        model_ = resnet18(num_classes = ops.num_classes,img_size=ops.img_size[0])
    elif ops.model == 'resnet_34':
        model_ = resnet34(num_classes = ops.num_classes,img_size=ops.img_size[0])
    elif ops.model == 'resnet_101':
        model_ = resnet101(num_classes = ops.num_classes,img_size=ops.img_size[0])
    elif ops.model == "squeezenet1_0":
        model_ = squeezenet1_0(num_classes=ops.num_classes)
    elif ops.model == "squeezenet1_1":
        model_ = squeezenet1_1(num_classes=ops.num_classes)
    elif ops.model == "shufflenetv2":
        model_ = ShuffleNetV2(ratio=1., num_classes=ops.num_classes)
    elif ops.model == "shufflenet_v2_x1_5":
        model_ = shufflenet_v2_x1_5(pretrained=False,num_classes=ops.num_classes)
    elif ops.model == "shufflenet_v2_x1_0":
        model_ = shufflenet_v2_x1_0(pretrained=False,num_classes=ops.num_classes)
    elif ops.model == "shufflenet_v2_x2_0":
        model_ = shufflenet_v2_x2_0(pretrained=False,num_classes=ops.num_classes)
    elif ops.model == "shufflenet":
        model_ = ShuffleNet(num_blocks = [2,4,2], num_classes=ops.num_classes, groups=3)
    elif ops.model == "mobilenetv2":
        model_ = MobileNetV2(num_classes=ops.num_classes)
    elif ops.model == "ReXNetV1":
        model_ = ReXNetV1(num_classes=ops.num_classes)
    elif ops.model == "test":
    ###########################################################################################################################

        print("test模型")
        # hand_joint_x, hand_joint_y, idx = test_hand_video.test_hhh(ops.model_path, 'pose_deploy.prototxt', 'pose_iter_102000.caffemodel', test_path, ops.vis)
        # ############## 计算角度变化 #################
        # ang_1 = angel_jisuan(flag_ang_i=5,flag_ang_j=6, hand_joint_x=hand_joint_x, hand_joint_y=hand_joint_y, idx=idx)
        # ang_2 = angel_jisuan(flag_ang_i=6,flag_ang_j=7, hand_joint_x=hand_joint_x, hand_joint_y=hand_joint_y, idx=idx)
        # ang_3 = angel_jisuan(flag_ang_i=7,flag_ang_j=8, hand_joint_x=hand_joint_x, hand_joint_y=hand_joint_y, idx=idx)
        # # plt.show()
        # ############# 求取红色区域位置坐标 ####################
        # red = red_shibie.red_j(ops.test_path)
        #
        # ############# 判断是否到达红色标记坐标 ###################
        # # radius = 319
        # radius = 25
        # neighbor_pixels = [[] for _ in range(idx)]
        # # 遍历以当前像素点为中心的周围区域
        # for p in range(0, idx):
        #     for i in range(int(hand_joint_x[p][8] - radius), int(hand_joint_x[p][8] + radius + 1)):
        #         for j in range(int(hand_joint_y[p][8] - radius), int(hand_joint_y[p][8] + radius + 1)):
        #             # 计算当前像素点到中心像素点的距离
        #             distance = ((i - hand_joint_x[p][8]) ** 2 + (j - hand_joint_y[p][8]) ** 2) ** 0.5
        #             # 如果距离小于等于半径，则将该像素点的坐标添加到列表中
        #             if distance <= radius:
        #                 neighbor_pixels[p].append((i, j))
        # # print(neighbor_pixels)
        #
        # flag_T = 0
        # # flag_F = 0
        # ten_or_more = []
        # for p in range(0, idx):
        #     for i in range(len(neighbor_pixels[p])):
        #         if neighbor_pixels[p][i] in red[p]:
        #             # if (117, 84) in red[i]:
        #             flag_T = 1
        #         # else:
        #         #     flag_F = 0
        #     if flag_T == 1:
        #         print("第%d张图片是否达到按钮位置：" % (p + 1))
        #         print('Ture')
        #         ten_or_more.append(flag_T)
        #     else:
        #         print("第%d张图片是否达到按钮位置：" % (p + 1))
        #         print('False')
        #         ten_or_more.append(flag_T)
        #     flag_T = 0
        #
        # xiaohao_time = 0
        # for i in range(len(ten_or_more)):
        #     if ten_or_more[i] and ten_or_more[i + 1] and ten_or_more[i + 2] and ten_or_more[i + 3] and ten_or_more[
        #         i + 4] and ten_or_more[i + 5] and ten_or_more[i + 6] and ten_or_more[i + 7] and ten_or_more[i + 8] and \
        #             ten_or_more[i + 9] == 1:
        #         xiaohao_time = frame_time * (i + 1)
        #         print("抵达按钮消耗时间: %f s" % xiaohao_time)
        #         break
        # if xiaohao_time == 0:
        #     print("未抵达按钮")
        #
        # print('well done ')
        #
        # sys.exit()  # 终止程序执行，不再执行以下代码

    ###########################################################################################################################
    elif ops.model == "mediapipe":
        print("mediapipe模型")
        hand_joint_x, hand_joint_y, idx = mediapipe_video.med_model(ops.model_path, test_video_path, ops.vis, clear_path="D:/python_project/handpose_shibie/handpose_x-main/shibie_img_med/")
        hand_joint_x_1, hand_joint_y_1, idx_1 = mediapipe_video.med_model(ops.model_path, test_video_path_sizhi, ops.vis, clear_path="D:/python_project/handpose_shibie/handpose_x-main/shibie_img_med_sizhi/")
        hand_joint_x_2, hand_joint_y_2, idx_2 = mediapipe_video.med_model(ops.model_path, test_video_path_muzhi,ops.vis, clear_path="D:/python_project/handpose_shibie/handpose_x-main/shibie_img_med_muzhi/")
        hand_joint_x_mu, hand_joint_y_mu, idx_mu = mediapipe_video.med_model(ops.model_path, test_video_path_muzhi_1, ops.vis,
                                                                    clear_path="D:/python_project/handpose_shibie/handpose_x-main/shibie_img_med_mu/")
        hand_joint_x_zhong, hand_joint_y_zhong, idx_zhong = mediapipe_video.med_model(ops.model_path, test_video_path_zhongzhi_1, ops.vis,
                                                                    clear_path="D:/python_project/handpose_shibie/handpose_x-main/shibie_img_med_zhong/")
        hand_joint_x_huan, hand_joint_y_huan, idx_huan = mediapipe_video.med_model(ops.model_path, test_video_path_huanzhi_1, ops.vis,
                                                                    clear_path="D:/python_project/handpose_shibie/handpose_x-main/shibie_img_med_huan/")
        hand_joint_x_xiao, hand_joint_y_xiao, idx_xiao = mediapipe_video.med_model(ops.model_path, test_video_path_xiaozhi_1, ops.vis,
                                                                    clear_path="D:/python_project/handpose_shibie/handpose_x-main/shibie_img_med_xiao/")
        # print("x坐标")
        # print(hand_joint_x)
        # print("y坐标")
        # print(hand_joint_y)
        ############## 计算角度变化 #################
        #################### SHE评估 #################
        socre_2_3 = angel_jisuan(flag_ang_i=2, flag_ang_j=3, hand_joint_x=hand_joint_x_2, hand_joint_y=hand_joint_y_2,
                               idx=idx_2)
        socre_3_4 = angel_jisuan(flag_ang_i=3, flag_ang_j=4, hand_joint_x=hand_joint_x_2, hand_joint_y=hand_joint_y_2,
                               idx=idx_2)
        socre_5_6 = angel_jisuan(flag_ang_i=5, flag_ang_j=6, hand_joint_x=hand_joint_x_1, hand_joint_y=hand_joint_y_1,
                               idx=idx_1)
        socre_6_7 = angel_jisuan(flag_ang_i=6, flag_ang_j=7, hand_joint_x=hand_joint_x_1, hand_joint_y=hand_joint_y_1,
                             idx=idx_1)
        socre_7_8 = angel_jisuan(flag_ang_i=7, flag_ang_j=8, hand_joint_x=hand_joint_x_1, hand_joint_y=hand_joint_y_1,
                             idx=idx_1)
        socre_9_10 = angel_jisuan(flag_ang_i=9, flag_ang_j=10, hand_joint_x=hand_joint_x_1, hand_joint_y=hand_joint_y_1,
                               idx=idx_1)
        socre_10_11 = angel_jisuan(flag_ang_i=10, flag_ang_j=11, hand_joint_x=hand_joint_x_1, hand_joint_y=hand_joint_y_1,
                               idx=idx_1)
        socre_11_12 = angel_jisuan(flag_ang_i=11, flag_ang_j=12, hand_joint_x=hand_joint_x_1, hand_joint_y=hand_joint_y_1,
                               idx=idx_1)
        socre_13_14 = angel_jisuan(flag_ang_i=13, flag_ang_j=14, hand_joint_x=hand_joint_x_1, hand_joint_y=hand_joint_y_1,
                                 idx=idx_1)
        socre_14_15 = angel_jisuan(flag_ang_i=14, flag_ang_j=15, hand_joint_x=hand_joint_x_1, hand_joint_y=hand_joint_y_1,
                                 idx=idx_1)
        socre_15_16 = angel_jisuan(flag_ang_i=15, flag_ang_j=16, hand_joint_x=hand_joint_x_1, hand_joint_y=hand_joint_y_1,
                                 idx=idx_1)
        socre_17_18 = angel_jisuan(flag_ang_i=17, flag_ang_j=18, hand_joint_x=hand_joint_x_1, hand_joint_y=hand_joint_y_1,
                                 idx=idx_1)
        socre_18_19 = angel_jisuan(flag_ang_i=18, flag_ang_j=19, hand_joint_x=hand_joint_x_1, hand_joint_y=hand_joint_y_1,
                                 idx=idx_1)
        socre_19_20 = angel_jisuan(flag_ang_i=19, flag_ang_j=20, hand_joint_x=hand_joint_x_1, hand_joint_y=hand_joint_y_1,
                                 idx=idx_1)

        socre_SIE = jisuan_SIE(socre_2_3, socre_3_4, socre_5_6, socre_6_7, socre_7_8, socre_9_10, socre_10_11, socre_11_12, socre_13_14, socre_14_15, socre_15_16,
                               socre_17_18, socre_18_19, socre_19_20)


        # plt.show()
        ############# 求取红色区域位置坐标 ####################
        red_shi = red_shibie.red_j(ops.test_path_shi)
        blue_shi = red_shibie.blue_j(ops.test_path_shi)

        red_mu = red_shibie.red_j(ops.test_path_mu)
        blue_mu = red_shibie.blue_j(ops.test_path_mu)

        red_zhong = red_shibie.red_j(ops.test_path_zhong)
        blue_zhong = red_shibie.blue_j(ops.test_path_zhong)

        red_huan = red_shibie.red_j(ops.test_path_huan)
        blue_huan = red_shibie.blue_j(ops.test_path_huan)

        red_xiao = red_shibie.red_j(ops.test_path_xiao)
        blue_xiao = red_shibie.blue_j(ops.test_path_xiao)

        # ############# 判断是否到达红色标记坐标 ###################
        # # radius = 319
        # radius = 10
        # neighbor_pixels = [[] for _ in range(idx)]
        # # 遍历以当前像素点为中心的周围区域
        # for p in range(0, idx):
        #     for i in range(int(hand_joint_x[p][8] - radius), int(hand_joint_x[p][8] + radius + 1)):
        #         for j in range(int(hand_joint_y[p][8] - radius), int(hand_joint_y[p][8] + radius + 1)):
        #             # 计算当前像素点到中心像素点的距离
        #             distance = ((i - hand_joint_x[p][8]) ** 2 + (j - hand_joint_y[p][8]) ** 2) ** 0.5
        #             # 如果距离小于等于半径，则将该像素点的坐标添加到列表中
        #             if distance <= radius:
        #                 neighbor_pixels[p].append((i, j))
        # # print(neighbor_pixels)
        #
        # flag_T = 0
        # # flag_F = 0
        # ten_or_more = []
        # for p in range(0, idx):
        #     for i in range(len(neighbor_pixels[p])):
        #         if neighbor_pixels[p][i] in red[p]:
        #             # if (117, 84) in red[i]:
        #             flag_T = 1
        #         # else:
        #         #     flag_F = 0
        #     if flag_T == 1:
        #         print("第%d张图片是否达到按钮位置：" % (p + 1))
        #         print('Ture')
        #         ten_or_more.append(flag_T)
        #     else:
        #         print("第%d张图片是否达到按钮位置：" % (p + 1))
        #         print('False')
        #         ten_or_more.append(flag_T)
        #     flag_T = 0
        #
        # xiaohao_time = 0
        # for i in range(len(ten_or_more)):
        #     if ten_or_more[i] and ten_or_more[i + 1] and ten_or_more[i + 2] and ten_or_more[i + 3] and ten_or_more[
        #         i + 4] and ten_or_more[i + 5] and ten_or_more[i + 6] and ten_or_more[i + 7] and ten_or_more[i + 8] and \
        #             ten_or_more[i + 9] and ten_or_more[i + 10] and ten_or_more[i + 11] and ten_or_more[i + 12]== 1:
        #         xiaohao_time = frame_time * (i + 1)
        #         print("抵达按钮消耗时间: %f s" % xiaohao_time)
        #         break
        # if xiaohao_time == 0:
        #     print("未抵达按钮")
        print("食指：")
        time_8_red, time_8_blue = hongse(idx, hand_joint_x, hand_joint_y, red_shi, blue_shi, frame_time, 8)
        print("拇指：")
        time_4_red, time_4_blue = hongse(idx_mu, hand_joint_x_mu, hand_joint_y_mu, red_mu, blue_mu, frame_time, 4)
        print("中指：")
        time_12_red, time_12_blue = hongse(idx_zhong, hand_joint_x_zhong, hand_joint_y_zhong, red_zhong, blue_zhong, frame_time, 12)
        print("环指：")
        time_16_red, time_16_blue = hongse(idx_huan, hand_joint_x_huan, hand_joint_y_huan, red_huan, blue_huan, frame_time, 16)
        print("小指：")
        time_20_red, time_20_blue = hongse(idx_xiao, hand_joint_x_xiao, hand_joint_y_xiao, red_xiao, blue_xiao, frame_time, 20)

        socre_touch = touchs(time_8_red, time_8_blue, time_4_red, time_4_blue, time_12_red, time_12_blue, time_16_red, time_16_blue, time_20_red, time_20_blue)

        ########## 总分数 #########
        zong_socre = socre_SIE*0.7 + socre_touch*0.3
        print("SIE分数: %f" % socre_SIE)
        print("按触分数: %f" % socre_touch)
        print("总分数：%f" % zong_socre)


        print('well done ')

        sys.exit()  # 终止程序执行，不再执行以下代码

    ###########################################################################################################################


























    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda:0" if use_cuda else "cpu")
    model_ = model_.to(device)
    model_.eval() # 设置为前向推断模式

    # print(model_)# 打印模型结构

    # 加载测试模型
    if os.access(ops.model_path,os.F_OK):# checkpoint
        chkpt = torch.load(ops.model_path, map_location=device)
        model_.load_state_dict(chkpt)
        print('load test model : {}'.format(ops.model_path))

    #---------------------------------------------------------------- 预测图片
    '''建议 检测手bbox后，crop手图片的预处理方式：
    # img 为原图
    x_min,y_min,x_max,y_max,score = bbox
    w_ = max(abs(x_max-x_min),abs(y_max-y_min))

    w_ = w_*1.1

    x_mid = (x_max+x_min)/2
    y_mid = (y_max+y_min)/2

    x1,y1,x2,y2 = int(x_mid-w_/2),int(y_mid-w_/2),int(x_mid+w_/2),int(y_mid+w_/2)

    x1 = np.clip(x1,0,img.shape[1]-1)
    x2 = np.clip(x2,0,img.shape[1]-1)

    y1 = np.clip(y1,0,img.shape[0]-1)
    y2 = np.clip(y2,0,img.shape[0]-1)
    '''
    with torch.no_grad():
        idx = 0
        hand_joint_x = []
        hand_joint_y = []
        for file in sorted(os.listdir(ops.test_path), key=extract_number):
            if '.jpg' not in file:
                continue
            print(file)
            idx += 1
            print('{}) image : {}'.format(idx,file))
            img = cv2.imread(ops.test_path + file)
            img_width = img.shape[1]
            img_height = img.shape[0]
            # 输入图片预处理
            img_ = cv2.resize(img, (ops.img_size[1],ops.img_size[0]), interpolation = cv2.INTER_CUBIC)
            img_ = img_.astype(np.float32)
            img_ = (img_-128.)/256.

            img_ = img_.transpose(2, 0, 1)
            img_ = torch.from_numpy(img_)
            img_ = img_.unsqueeze_(0)

            if use_cuda:
                img_ = img_.cuda()  # (bs, 3, h, w)
            pre_ = model_(img_.float()) # 模型推理
            output = pre_.cpu().detach().numpy()
            output = np.squeeze(output)

            pts_hand = {} #构建关键点连线可视化结构
            for i in range(int(output.shape[0]/2)):
                x = (output[i*2+0]*float(img_width))
                y = (output[i*2+1]*float(img_height))

                pts_hand[str(i)] = {}
                pts_hand[str(i)] = {
                    "x":x,
                    "y":y,
                    }
            draw_bd_handpose(img,pts_hand,0,0) # 绘制关键点连线

            h_j_x = []
            h_j_y = []

            print("\n第%d张图片数据：\n" % idx)
            #------------- 绘制关键点
            for i in range(int(output.shape[0]/2)):
                x = (output[i*2+0]*float(img_width))
                y = (output[i*2+1]*float(img_height))

                print("  第%d关节的数据" % (i+1))
                print(x)
                print(y)
                h_j_x.append(x)
                h_j_y.append(y)
                cv2.circle(img, (int(x), int(y)), 3, (255, 50, 60), -1)
                cv2.circle(img, (int(x), int(y)), 1, (255, 150, 180), -1)

            if ops.vis:
                cv2.namedWindow('image', cv2.WINDOW_KEEPRATIO)
                cv2.imshow('image',img)
                # if cv2.waitKey(600) == 27 :
                if cv2.waitKey(33) == 27:
                    break

            hand_joint_x.append(h_j_x)
            hand_joint_y.append(h_j_y)

    print(hand_joint_x)
    print(hand_joint_y)

    cv2.destroyAllWindows()

    ############## 计算角度变化 #################
    ang_1 = angel_jisuan(flag_ang_i=5, flag_ang_j=6, hand_joint_x=hand_joint_x, hand_joint_y=hand_joint_y, idx=idx)
    ang_2 = angel_jisuan(flag_ang_i=6, flag_ang_j=7, hand_joint_x=hand_joint_x, hand_joint_y=hand_joint_y, idx=idx)
    ang_3 = angel_jisuan(flag_ang_i=7, flag_ang_j=8, hand_joint_x=hand_joint_x, hand_joint_y=hand_joint_y, idx=idx)
    # plt.show()
    ############# 求取红色区域位置坐标 ####################
    red = red_shibie.red_j(ops.test_path)
    # print("红色区域坐标：")
    # print(red)
    # print(idx)

    # ############# 扩大红色像素点范围 ###################
    # # 定义像素点的周围范围
    # neighborhood_size = 100
    #
    # # 遍历所有像素点
    # red_beifen = red
    # for p in range(len(red)):
    #     print("更新前:")
    #     print(len(red[p]))
    #     for i in range(0, img_width):
    #         for j in range(0, img_height):
    #             # 如果当前像素点是白色
    #             if (i, j) in red_beifen[p]:
    #                 # 遍历当前像素点周围的像素点
    #                 for k in range(max(0, i - neighborhood_size), min(img_width, i + neighborhood_size + 1)):
    #                     for l in range(max(0, j - neighborhood_size), min(img_height, j + neighborhood_size + 1)):
    #                         # 将周围像素点设置为白色
    #                         kk = (k, l)
    #                         red[p].append(kk)
    #     print("更新后:")
    #     print(len(red[p]))





    ############# 判断是否到达红色标记坐标 ###################
    # radius = 319
    radius = 5
    neighbor_pixels = [[] for _ in range(idx)]
    # 遍历以当前像素点为中心的周围区域
    for p in range(0, idx):
        for i in range(int(hand_joint_x[p][8] - radius), int(hand_joint_x[p][8] + radius + 1)):
            for j in range(int(hand_joint_y[p][8] - radius), int(hand_joint_y[p][8] + radius + 1)):
                # 计算当前像素点到中心像素点的距离
                distance = ((i - hand_joint_x[p][8]) ** 2 + (j - hand_joint_y[p][8]) ** 2) ** 0.5
                # 如果距离小于等于半径，则将该像素点的坐标添加到列表中
                if distance <= radius:
                    neighbor_pixels[p].append((i, j))
    # print(neighbor_pixels)

    flag_T = 0
    # flag_F = 0
    ten_or_more = []
    for p in range(0, idx):
        for i in range(len(neighbor_pixels[p])):
            if neighbor_pixels[p][i] in red[p]:
                # if (117, 84) in red[i]:
                flag_T = 1
            # else:
            #     flag_F = 0
        if flag_T == 1:
            print("第%d张图片是否达到按钮位置：" % (p + 1))
            print('Ture')
            ten_or_more.append(flag_T)
        else:
            print("第%d张图片是否达到按钮位置：" % (p + 1))
            print('False')
            ten_or_more.append(flag_T)
        flag_T = 0

    xiaohao_time = 0
    for i in range(len(ten_or_more)):
        if ten_or_more[i] and ten_or_more[i+1] and ten_or_more[i+2] and ten_or_more[i+3] and ten_or_more[i+4] and ten_or_more[i+5] and ten_or_more[i+6] and ten_or_more[i+7] and ten_or_more[i+8] and ten_or_more[i+9] == 1:
            xiaohao_time = frame_time*(i+1)
            print("抵达按钮消耗时间: %f s" % xiaohao_time)
            break
    if xiaohao_time == 0:
        print("未抵达按钮")


    print('well done ')
