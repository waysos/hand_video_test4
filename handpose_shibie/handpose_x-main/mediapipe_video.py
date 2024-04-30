import mediapipe as mp
import cv2
import os

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





def med_model(model_path, test_video_path, vis, clear_path):
    ############# 基础设置 ####################
    model_path = model_path
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    # clear_folder("D:/python_project/handpose_shibie/handpose_x-main/shibie_img_med/")
    clear_folder(clear_path)
    # Create a hand landmarker instance with the video mode:
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),  # 模型路径
        running_mode=VisionRunningMode.VIDEO,  # 输入形式
        num_hands=1,  # 手的数量
        min_hand_detection_confidence=0.5,  # 手部检测（非标记）的最低置信度（0-1）
        min_hand_presence_confidence=0.5,  # 标记手部关键点的最低置信度（0-1）
        min_tracking_confidence=0.5,  # 手部追踪被视为成功的最低置信度（0-1）
    )

    ########## 准备数据 ################
    # 读取视频文件
    video_file = test_video_path
    cap = cv2.VideoCapture(video_file)

    frames = []
    timestamps = []

    # 逐帧读取视频并将每帧转换为numpy数组
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # 获取时间戳
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
        # 将帧转换为RGB格式（如果需要）
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        timestamps.append(timestamp)

    # 关闭视频文件
    cap.release()
    # print(frames)
    # frames列表现在包含视频的所有帧，每个帧都是一个numpy数组
    handpose_xs = []
    handpose_ys = []

    for i in range(len(frames)):
        numpy_frame_from_opencv = frames[i]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_frame_from_opencv)
        hand_landmarker_results = []
        with HandLandmarker.create_from_options(options) as landmarker:
            frame_timestamp_ms = int(timestamps[i])
            hand_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
            # print("第%d帧图像的21个关键点坐标" % (i+1))
            # print(hand_landmarker_result.hand_landmarks)
            hand_landmarker_results.append(
                hand_landmarker_result.hand_landmarks)  # hand_landmarker_result.hand_landmarks为21 个手部标志，每个标志由x,y和z坐标组成。x和y坐标分别通过图像宽度和高度标准化为[0.0,1.0]。z坐标代表地标深度,以手腕处的深度为原点。值越小，地标距离相机越近。z的大小使用与x大致相同的比例。
            hand_mark = hand_landmarker_result.hand_landmarks
            if hand_mark != []:
                hand_mark_old = hand_mark
            # print("第%d帧图像的第1个关节的x归一化坐标:" % (i+1))
            # print(hand_mark[0][0].x) # 之所以是[0][0]，是因为输出的是个两层列表，第一层只有一维，第一个[0]表示选取全部的关键点对象
            # print(hand_mark)
            handpose_x = []
            handpose_y = []
            for j in range(0, 21):
                if hand_mark != []:
                    handpose_x.append(hand_mark[0][j].x)
                    handpose_y.append(hand_mark[0][j].y)
                else:
                    handpose_x.append(hand_mark_old[0][j].x)
                    handpose_y.append(hand_mark_old[0][j].y)


            handpose_xs.append(handpose_x)
            handpose_ys.append(handpose_y)

    # print(handpose_xs)
    # print(handpose_ys)

    ####################### 绘制图像 #########################
    hui_frames = []
    hand_pose_x = []
    hand_pose_y = []
    cap = cv2.VideoCapture(video_file)
    idx = 0
    # 逐帧读取视频并处理
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # hui_frames.append(frame)
        # 绘制检测到的手部关键点和连接线
        height, width, _ = frame.shape
        hand_points_x = handpose_xs[idx]
        hand_points_y = handpose_ys[idx]
        hand_points = []
        hand_pose_x_z = []
        hand_pose_y_z = []
        for landmark in range(len(hand_points_x)):
            cx, cy = int(hand_points_x[landmark] * width), int(hand_points_y[landmark] * height)
            hand_pose_x_z.append(cx)
            hand_pose_y_z.append(cy)
            hand_points.append((cx, cy))
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
        hand_pose_x.append(hand_pose_x_z)
        hand_pose_y.append(hand_pose_y_z)

        pts_hand = {}  # 构建关键点连线可视化结构
        for i in range(len(hand_points_x)):
            x = int(hand_points_x[i] * width)
            y = int(hand_points_y[i] * height)

            pts_hand[str(i)] = {}
            pts_hand[str(i)] = {
                "x": x,
                "y": y,
            }
        draw_bd_handpose(frame, pts_hand, 0, 0)  # 绘制关键点连线
        cv2.namedWindow('Hand Landmarks Detection', cv2.WINDOW_KEEPRATIO)  # 可以调整窗口大小
        # 设置保存地址
        save_folder = (clear_path)

        # 在适当的位置添加保存图像的代码
        cv2.imwrite(save_folder + 'hand_landmarks_frame_{}.png'.format(idx), frame)
        if vis:
            cv2.imshow('Hand Landmarks Detection', frame)
            # cv2.resizeWindow('Hand Landmarks Detection', 1080, 1920)  # 设置窗口大小
        idx += 1
        # 按下 'q' 键退出
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break



    # 释放视频文件并关闭窗口
    cap.release()
    cv2.destroyAllWindows()
    return hand_pose_x, hand_pose_y, idx













