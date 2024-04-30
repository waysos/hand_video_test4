#-*-coding:utf-8-*-
# date:2021-04-5
# Author: Eric.Lee
# function: Inference

import os
import argparse
import torch
import torch.nn as nn
import numpy as np

import time
import datetime
import os
import math
from datetime import datetime
import cv2
import torch.nn.functional as F
import matplotlib.pyplot as plt
import red_shibie
import chaifen_shipin

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
def angel_jisuan(flag_ang, hand_joint_x, hand_joint_y, idx):
    if flag_ang == 8.7:
        angle_8_7 = []
        deta_x = 0
        deta_y = 0
        angle_rad_8_7 = []
        for i in range(0, idx):
            deta_x = hand_joint_x[i][7] - hand_joint_x[i][8]
            deta_y = hand_joint_y[i][7] - hand_joint_y[i][8]
            angle_rad_8_7.append(math.atan(deta_y / deta_x))
            angle_8_7.append(math.degrees(angle_rad_8_7[i]))

        print("第八个和第七个关节形成角度：")
        print(angle_rad_8_7)
        img_x = [i for i in range(1, idx+1)]

        # plt.rcParams['font.family'] = ['SimHei']  # 设置中文字体为黑体，可以根据需要更改为其他字体
        plt.rcParams['font.family'] = ['Microsoft YaHei']  # 微软雅黑
        plt.figure(1)
        plt.plot(img_x, angle_8_7)
        plt.title('第八个和第七个关节形成角度')
        plt.xlabel('img_x')
        plt.ylabel('angle_8_7')
        plt.savefig('./picture/hand_ang_8_7.png')
        # plt.show()

        output = angle_rad_8_7
    elif flag_ang == 7.6:
        angle_7_6 = []
        deta_x = 0
        deta_y = 0
        angle_rad_7_6 = []
        for i in range(0, idx):
            deta_x = hand_joint_x[i][6] - hand_joint_x[i][7]
            deta_y = hand_joint_y[i][6] - hand_joint_y[i][7]
            angle_rad_7_6.append(math.atan(deta_y / deta_x))
            angle_7_6.append(math.degrees(angle_rad_7_6[i]))

        print("第七个和第六个关节形成角度：")
        print(angle_rad_7_6)
        img_x = [i for i in range(1, idx + 1)]

        # plt.rcParams['font.family'] = ['SimHei']  # 设置中文字体为黑体，可以根据需要更改为其他字体
        plt.rcParams['font.family'] = ['Microsoft YaHei']  # 微软雅黑
        plt.figure(2)
        plt.plot(img_x, angle_7_6)
        plt.title('第七个和第六个关节形成角度')
        plt.xlabel('img_x')
        plt.ylabel('angle_8_7')
        plt.savefig('./picture/hand_ang_7_6.png')
        # plt.show()

        output = angle_rad_7_6
    elif flag_ang == 6.5:
        angle = []
        deta_x = 0
        deta_y = 0
        angle_rad = []
        for i in range(0, idx):
            deta_x = hand_joint_x[i][5] - hand_joint_x[i][6]
            deta_y = hand_joint_y[i][5] - hand_joint_y[i][6]
            angle_rad.append(math.atan(deta_y / deta_x))
            angle.append(math.degrees(angle_rad[i]))

        print("第六个和第五个关节形成角度：")
        print(angle_rad)
        img_x = [i for i in range(1, idx + 1)]

        # plt.rcParams['font.family'] = ['SimHei']  # 设置中文字体为黑体，可以根据需要更改为其他字体
        # plt.rcParams['font.family'] = ['Arial']
        plt.rcParams['font.family'] = ['Microsoft YaHei']  # 微软雅黑
        plt.figure(3)
        plt.plot(img_x, angle)
        plt.title('第六个和第五个关节形成角度')
        plt.xlabel('img_x')
        plt.ylabel('angle')
        plt.savefig('./picture/hand_ang_6_5.png')
        # plt.show()

        output = angle_rad
    return output

def extract_number(filename):
    # 从文件名中提取数字部分
    return int(''.join(filter(str.isdigit, filename)))


if __name__ == "__main__":



    parser = argparse.ArgumentParser(description=' Project Hand Pose Inference')
    output_img_path = "D:/python_project/handpose_shibie/handpose_x-main/shibie_img_ResNet/"
    clear_folder(output_img_path)
    # parser.add_argument('--model_path', type=str, default = './weights/ReXNetV1-size-256-wingloss102-0.122.pth',
    #     help = 'model_path') # 模型路径
    parser.add_argument('--model_path', type=str, default = 'D:/hand_data/handpose_x_model/resnet_50-size-256-loss-0.0642.pth',
        help = 'model_path') # 模型路径
    parser.add_argument('--test_video_path', type=str,
                        default='D:/hand_data/test_video/test_5.mp4',
                        help='test_video_path')  # 测试视频路径
    parser.add_argument('--model', type=str, default = 'resnet_50',
        help = '''model : resnet_34,resnet_50,resnet_101,squeezenet1_0,squeezenet1_1,shufflenetv2,shufflenet,mobilenetv2
            shufflenet_v2_x1_5 ,shufflenet_v2_x1_0 , shufflenet_v2_x2_0,ReXNetV1''') # 模型类型
    parser.add_argument('--num_classes', type=int , default = 42,
        help = 'num_classes') #  手部21关键点， (x,y)*2 = 42
    parser.add_argument('--GPUS', type=str, default = '0',
        help = 'GPUS') # GPU选择
    parser.add_argument('--test_path', type=str, default = './image/',
        help = 'test_path') # 测试图片路径
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

    test_path =  ops.test_path # 测试图片文件夹路径
    test_video_path = ops.test_video_path # 测试视频路径


    ###################### 将视频文件转为一秒三十帧的图片们 ###########################
    video_path = test_video_path
    output_dir = test_path
    frame_time = chaifen_shipin.split_video_to_frames(video_path, output_dir)



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

            # print("\n第%d张图片数据：\n" % idx)
            #------------- 绘制关键点
            for i in range(int(output.shape[0]/2)):
                x = (output[i*2+0]*float(img_width))
                y = (output[i*2+1]*float(img_height))

                # print("  第%d关节的数据" % (i+1))
                # print(x)
                # print(y)
                h_j_x.append(x)
                h_j_y.append(y)
                cv2.circle(img, (int(x), int(y)), 3, (255, 50, 60), -1)
                cv2.circle(img, (int(x), int(y)), 1, (255, 150, 180), -1)

                save_folder = (output_img_path)

                # 在适当的位置添加保存图像的代码
                cv2.imwrite(save_folder + 'hand_landmarks_frame_{}.png'.format(idx), img)

            if ops.vis:
                cv2.namedWindow('image', cv2.WINDOW_KEEPRATIO)  # 可以调整窗口大小
                cv2.namedWindow('image',0)
                cv2.imshow('image',img)
                # if cv2.waitKey(600) == 27 :
                if cv2.waitKey(10) == 27:
                    break

            hand_joint_x.append(h_j_x)
            hand_joint_y.append(h_j_y)

    print(hand_joint_x)
    print(hand_joint_y)

    cv2.destroyAllWindows()

    ############## 计算角度变化 #################
    ang_1 = angel_jisuan(flag_ang=8.7, hand_joint_x=hand_joint_x, hand_joint_y=hand_joint_y, idx=idx)
    ang_2 = angel_jisuan(flag_ang=7.6, hand_joint_x=hand_joint_x, hand_joint_y=hand_joint_y, idx=idx)
    ang_3 = angel_jisuan(flag_ang=6.5, hand_joint_x=hand_joint_x, hand_joint_y=hand_joint_y, idx=idx)
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
    radius = 10
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
