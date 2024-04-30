import os
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
    print(pts_t_z_tensor)

    return pts_t_z_tensor, imgs

def y_model_output(ops):
    if ops.model == 'resnet_50':

        model_ = resnet50(pretrained=True, num_classes=ops.num_classes, img_size=ops.img_size[0],
                          dropout_factor=ops.dropout)
    elif ops.model == 'resnet_18':
        model_ = resnet18(pretrained=True, num_classes=ops.num_classes, img_size=ops.img_size[0],
                          dropout_factor=ops.dropout)
    elif ops.model == 'resnet_34':
        model_ = resnet34(pretrained=True, num_classes=ops.num_classes, img_size=ops.img_size[0],
                          dropout_factor=ops.dropout)
    elif ops.model == 'resnet_101':
        model_ = resnet101(pretrained=True, num_classes=ops.num_classes, img_size=ops.img_size[0],
                           dropout_factor=ops.dropout)
    elif ops.model == "squeezenet1_0":
        model_ = squeezenet1_0(pretrained=True, num_classes=ops.num_classes, dropout_factor=ops.dropout)
    elif ops.model == "squeezenet1_1":
        model_ = squeezenet1_1(pretrained=True, num_classes=ops.num_classes, dropout_factor=ops.dropout)
    elif ops.model == "shufflenetv2":
        model_ = ShuffleNetV2(ratio=1., num_classes=ops.num_classes, dropout_factor=ops.dropout)
    elif ops.model == "shufflenet_v2_x1_5":
        model_ = shufflenet_v2_x1_5(pretrained=False, num_classes=ops.num_classes)
    elif ops.model == "shufflenet_v2_x1_0":
        model_ = shufflenet_v2_x1_0(pretrained=False, num_classes=ops.num_classes)
    elif ops.model == "shufflenet_v2_x2_0":
        model_ = shufflenet_v2_x2_0(pretrained=False, num_classes=ops.num_classes)
    elif ops.model == "shufflenet":
        model_ = ShuffleNet(num_blocks=[2, 4, 2], num_classes=ops.num_classes, groups=3, dropout_factor=ops.dropout)
    elif ops.model == "mobilenetv2":
        model_ = MobileNetV2(num_classes=ops.num_classes, dropout_factor=ops.dropout)
    elif ops.model == "ReXNetV1":
        model_ = ReXNetV1(num_classes=ops.num_classes, dropout_factor=ops.dropout)

    else:
        print(" no support the model")
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model_ = model_.to(device)
    fintune_model = ops.fintune_model
    chkpt = torch.load(fintune_model, map_location=device)  # 将预训练模型参数放到chkpt
    model_.load_state_dict(chkpt)  # 加载预训练模型参数
    print('load fintune model : {}'.format(ops.fintune_model))
    if ops.loss_define != 'wing_loss':
        criterion = nn.MSELoss(reduce=True, reduction='mean')
    dataset = LoadImagesAndLabels(ops=ops, img_size=ops.img_size, flag_agu=ops.flag_agu, fix_res=ops.fix_res, vis=False)
    dataloader = DataLoader(dataset,
                            batch_size=ops.batch_size,
                            num_workers=ops.num_workers,
                            shuffle=True,
                            pin_memory=False,
                            drop_last=True)
    loss_mean = 0.  # 损失均值
    loss_idx = 0.  # 损失计算计数器
    for i, (imgs_, pts_) in enumerate(dataloader):
        # print('imgs_, pts_',imgs_.size(), pts_.size())
        if use_cuda:
            imgs_ = imgs_.cuda()  # pytorch 的 数据输入格式 ： (batch, channel, height, width)
            pts_ = pts_.cuda()

        output = model_(imgs_.float())  # 将图片按浮点数输入（确保与模型的输入类型匹配），模型得出输出
        # print("*************************输出为：")
        # print(output)
        if ops.loss_define == 'wing_loss':
            loss = got_total_wing_loss(output, pts_.float())  # 通过输出和实际标注点pts得出损失
        else:
            loss = criterion(output, pts_.float())  # 如果非"wing_loss"均方根误差MSE作为替代
        loss_mean += loss.item()  # 累加， 为了计算每个 epoch 中所有批次的平均损失值 loss_mean/loss_idx
        loss_idx += 1.
        if i % 10 == 0:
            loc_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print('Mean Loss : %.6f - Loss: %.6f'%(loss_mean/loss_idx,loss.item()))




if __name__ == '__main__':
    freeze_support()

    parser = argparse.ArgumentParser(description=' Project Hand Train')
    parser.add_argument('--seed', type=int, default=126673,
                        help='seed')  # 设置随机种子
    # parser.add_argument('--model_exp', type=str, default = './model_exp',
    #     help = 'model_exp') # 模型输出文件夹
    parser.add_argument('--model_exp', type=str, default='D:/hand_data/model_z',
                        help='model_exp')  # 模型输出文件夹D:/hand_data/model_z
    parser.add_argument('--model', type=str, default='squeezenet1_1',
                        help='''model : resnet_34,resnet_50,resnet_101,squeezenet1_0,squeezenet1_1,shufflenetv2,shufflenet,mobilenetv2
                shufflenet_v2_x1_5 ,shufflenet_v2_x1_0 , shufflenet_v2_x2_0,ReXNetV1''')  # 模型类型
    parser.add_argument('--num_classes', type=int, default=42,
                        help='num_classes')  # landmarks 个数*2
    parser.add_argument('--GPUS', type=str, default='0',
                        help='GPUS')  # GPU选择

    # parser.add_argument('--train_path', type=str,
    #     default = "./handpose_datasets_v1/",
    #     help = 'datasets')# 训练集标注信息
    parser.add_argument('--train_path', type=str,
                        default="D:/hand_data/handpose_datasets_v1-2021-01-31/handpose_datasets_v1/",
                        help='datasets')  # 训练集标注信息D:\hand_data\handpose_datasets_v1-2021-01-31

    parser.add_argument('--pretrained', type=bool, default=True,
                        help='imageNet_Pretrain')  # 初始化学习率
    # parser.add_argument('--fintune_model', type=str, default = 'None',
    #     help = 'fintune_model') # fintune model 预训练模型
    parser.add_argument('--fintune_model', type=str,
                        default='D:/hand_data/handpose_x_model/squeezenet1_1-size-256-loss-0.0732.pth',
                        help='fintune_model')  # fintune model 预训练模型
    parser.add_argument('--loss_define', type=str, default='wing_loss',
                        help='define_loss')  # 损失函数定义
    parser.add_argument('--init_lr', type=float, default=1e-3,
                        help='init learning Rate')  # 初始化学习率
    parser.add_argument('--lr_decay', type=float, default=0.1,
                        help='learningRate_decay')  # 学习率权重衰减率
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                        help='weight_decay')  # 优化器正则损失权重
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')  # 优化器动量
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch_size')  # 训练每批次图像数量
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout')  # dropout
    # parser.add_argument('--epochs', type=int, default = 3000,
    #     help = 'epochs') # 训练周期
    parser.add_argument('--epochs', type=int, default=5,
                        help='epochs')  # 训练周期
    # parser.add_argument('--num_workers', type=int, default = 10,
    #     help = 'num_workers') # 训练数据生成器线程数
    parser.add_argument('--num_workers', type=int, default=0,
                        help='num_workers')  # 训练数据生成器线程数
    parser.add_argument('--img_size', type=tuple, default=(256, 256),
                        help='img_size')  # 输入模型图片尺寸
    parser.add_argument('--flag_agu', type=bool, default=True,
                        help='data_augmentation')  # 训练数据生成器是否进行数据扩增
    parser.add_argument('--fix_res', type=bool, default=False,
                        help='fix_resolution')  # 输入模型样本图片是否保证图像分辨率的长宽比
    parser.add_argument('--clear_model_exp', type=bool, default=False,
                        help='clear_model_exp')  # 模型输出文件夹是否进行清除
    parser.add_argument('--log_flag', type=bool, default=False,
                        help='log flag')  # 是否保存训练 log

    # --------------------------------------------------------------------------
    args = parser.parse_args()  # 解析添加参数
    ops = args
    y_model_output(ops)
