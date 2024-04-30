import cv2
import time
import os
import numpy as np
import torch

def extract_number(filename):
    # 从文件名中提取数字部分
    return int(''.join(filter(str.isdigit, filename)))

def draw_bd_handpose(img_,hand_,x,y):
    thick = 2
    colors = [(0,215,255),(255,115,55),(5,255,55),(25,15,255),(225,15,55)]
    #
    cv2.line(img_, (int(hand_['0']['x']+x), int(hand_['0']['y']+y)),(int(hand_['1']['x']+x), int(hand_['1']['y']+y)), colors[0], thick)
    cv2.line(img_, (int(hand_['1']['x']+x), int(hand_['1']['y']+y)),(int(hand_['2']['x']+x), int(hand_['2']['y']+y)), colors[0], thick)
    cv2.line(img_, (int(hand_['2']['x']+x), int(hand_['2']['y']+y)),(int(hand_['3']['x']+x), int(hand_['3']['y']+y)), colors[0], thick)
    cv2.line(img_, (int(hand_['3']['x']+x), int(hand_['3']['y']+y)),(int(hand_['4']['x']+x), int(hand_['4']['y']+y)), colors[0], thick)

    cv2.line(img_, (int(hand_['0']['x']+x), int(hand_['0']['y']+y)),(int(hand_['5']['x']+x), int(hand_['5']['y']+y)), colors[1], thick)
    cv2.line(img_, (int(hand_['5']['x']+x), int(hand_['5']['y']+y)),(int(hand_['6']['x']+x), int(hand_['6']['y']+y)), colors[1], thick)
    cv2.line(img_, (int(hand_['6']['x']+x), int(hand_['6']['y']+y)),(int(hand_['7']['x']+x), int(hand_['7']['y']+y)), colors[1], thick)
    cv2.line(img_, (int(hand_['7']['x']+x), int(hand_['7']['y']+y)),(int(hand_['8']['x']+x), int(hand_['8']['y']+y)), colors[1], thick)

    cv2.line(img_, (int(hand_['0']['x']+x), int(hand_['0']['y']+y)),(int(hand_['9']['x']+x), int(hand_['9']['y']+y)), colors[2], thick)
    cv2.line(img_, (int(hand_['9']['x']+x), int(hand_['9']['y']+y)),(int(hand_['10']['x']+x), int(hand_['10']['y']+y)), colors[2], thick)
    cv2.line(img_, (int(hand_['10']['x']+x), int(hand_['10']['y']+y)),(int(hand_['11']['x']+x), int(hand_['11']['y']+y)), colors[2], thick)
    cv2.line(img_, (int(hand_['11']['x']+x), int(hand_['11']['y']+y)),(int(hand_['12']['x']+x), int(hand_['12']['y']+y)), colors[2], thick)

    cv2.line(img_, (int(hand_['0']['x']+x), int(hand_['0']['y']+y)),(int(hand_['13']['x']+x), int(hand_['13']['y']+y)), colors[3], thick)
    cv2.line(img_, (int(hand_['13']['x']+x), int(hand_['13']['y']+y)),(int(hand_['14']['x']+x), int(hand_['14']['y']+y)), colors[3], thick)
    cv2.line(img_, (int(hand_['14']['x']+x), int(hand_['14']['y']+y)),(int(hand_['15']['x']+x), int(hand_['15']['y']+y)), colors[3], thick)
    cv2.line(img_, (int(hand_['15']['x']+x), int(hand_['15']['y']+y)),(int(hand_['16']['x']+x), int(hand_['16']['y']+y)), colors[3], thick)

    cv2.line(img_, (int(hand_['0']['x']+x), int(hand_['0']['y']+y)),(int(hand_['17']['x']+x), int(hand_['17']['y']+y)), colors[4], thick)
    cv2.line(img_, (int(hand_['17']['x']+x), int(hand_['17']['y']+y)),(int(hand_['18']['x']+x), int(hand_['18']['y']+y)), colors[4], thick)
    cv2.line(img_, (int(hand_['18']['x']+x), int(hand_['18']['y']+y)),(int(hand_['19']['x']+x), int(hand_['19']['y']+y)), colors[4], thick)
    cv2.line(img_, (int(hand_['19']['x']+x), int(hand_['19']['y']+y)),(int(hand_['20']['x']+x), int(hand_['20']['y']+y)), colors[4], thick)



def test_hhh(path, txtname, model_filename, test_path, vis):
    protoFile = os.path.join(path, txtname) # 'pose_deploy.prototxt'  # Caffe模型文件
    weightsFile = os.path.join(path, model_filename) # 模型权重文件
    nPoints = 21
    POSE_PAIRS = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], [10, 11], [11, 12],
                  [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)


    with torch.no_grad():
        idx = 0
        hand_joint_x = []
        hand_joint_y = []

        for file in sorted(os.listdir(test_path), key=extract_number):
            if '.jpg' not in file:
                continue
            idx += 1

            frame = cv2.imread(test_path + file)
            frameCopy = np.copy(frame)
            frameWidth = frame.shape[1]
            frameHeight = frame.shape[0]
            aspect_ratio = frameWidth / frameHeight

            threshold = 0.1
            # if use_cuda:
            #     frame = frame.cuda()  # (bs, 3, h, w)

            t = time.time()
            # input image dimensions for the network
            inHeight = 368
            inWidth = int(((aspect_ratio * inHeight) * 8) // 8)
            inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)

            net.setInput(inpBlob)

            output = net.forward()
            print("time taken by network : {:.3f}".format(time.time() - t))

            # Empty list to store the detected keypoints
            points = []

            pts_hand = {}  # 构建关键点连线可视化结构
            for i in range(nPoints):
                probMap = output[0, i, :, :]
                probMap = cv2.resize(probMap, (frameWidth, frameHeight))

                # Find global maxima of the probMap.
                minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

                x = point[0]
                y = point[1]

                pts_hand[str(i)] = {}
                pts_hand[str(i)] = {
                    "x":x,
                    "y":y,
                    }
            draw_bd_handpose(frameCopy,pts_hand,0,0) # 绘制关键点连线

            h_j_x = []
            h_j_y = []

            print("\n第%d张图片数据：\n" % idx)
            for i in range(0, nPoints):
                # confidence map of corresponding body's part.
                probMap = output[0, i, :, :]
                probMap = cv2.resize(probMap, (frameWidth, frameHeight))

                # Find global maxima of the probMap.
                minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

                # if prob > threshold:
                #     print('tesstss')
                # if True:
                    # cv2.circle(frameCopy, (int(point[0]), int(point[1])), 8, (0, 255, 255), thickness=-1,
                    #            lineType=cv2.FILLED)
                    # cv2.putText(frameCopy, "{}".format(i), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    #             (0, 0, 255), 2, lineType=cv2.LINE_AA)
                    #
                    # # Add the point to the list if the probability is greater than the threshold
                    # points.append((int(point[0]), int(point[1])))
                print("  第%d关节的数据" % (i + 1))
                print(point[0])
                print(point[1])
                h_j_x.append(int(point[0]))
                h_j_y.append(int(point[1]))
                cv2.circle(frameCopy, (int(point[0]), int(point[1])), 3, (255, 50, 60), -1)
                cv2.circle(frameCopy, (int(point[0]), int(point[1])), 1, (255, 150, 180), -1)
            if vis:
                cv2.namedWindow('image', cv2.WINDOW_KEEPRATIO)
                cv2.imshow('image',frameCopy)
                # if cv2.waitKey(600) == 27 :
                if cv2.waitKey(1) == 27:
                    break

            hand_joint_x.append(h_j_x)
            hand_joint_y.append(h_j_y)
    print('x:')
    print(hand_joint_x)
    print('y:')
    print(hand_joint_y)
    cv2.destroyAllWindows()

    return hand_joint_x, hand_joint_y, idx




