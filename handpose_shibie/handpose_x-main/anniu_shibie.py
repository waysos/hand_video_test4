import mediapipe as mp
import cv2

def draw_juxing_name(img, origin_x, origin_y, width, height, name, color_num):
    thick = 2
    colors = [(0, 215, 255), (255, 115, 55), (5, 255, 55), (25, 15, 255), (225, 15, 55)]

    cv2.line(img, (int(origin_x), int(origin_y)), (int(origin_x + width), int(origin_y)), colors[color_num], thick)
    cv2.line(img, (int(origin_x + width), int(origin_y)), (int(origin_x + width), int(origin_y + height)), colors[color_num], thick)
    cv2.line(img, (int(origin_x + width), int(origin_y + height)), (int(origin_x), int(origin_y + height)), colors[color_num], thick)
    cv2.line(img, (int(origin_x), int(origin_y + height)), (int(origin_x), int(origin_y)), colors[color_num], thick)

    cv2.putText(img, name, (int(origin_x), int(origin_y + height)), cv2.FONT_HERSHEY_COMPLEX, 2.0, colors[color_num], thick)



def anniu():
    ############# 基础设置 ####################
    model_path = 'D:/hand_data/Madiapipe/Model/efficientdet_lite2.tflite'
    test_video_path = 'D:/hand_data/test_video/test_3.mp4'
    shibie_num = 5
    vis = True
    BaseOptions = mp.tasks.BaseOptions
    ObjectDetector = mp.tasks.vision.ObjectDetector
    ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    # Create a hand landmarker instance with the video mode:
    options = ObjectDetectorOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        max_results=shibie_num,
        running_mode=VisionRunningMode.VIDEO
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
        # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        timestamps.append(timestamp)

    # 关闭视频文件
    cap.release()
    # print(frames)
    # frames列表现在包含视频的所有帧，每个帧都是一个numpy数组
    detection_results = []
    for i in range(len(frames)):
        numpy_frame_from_opencv = frames[i]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_frame_from_opencv)
        hand_landmarker_results = []
        # with HandLandmarker.create_from_options(options) as landmarker:
        with ObjectDetector.create_from_options(options) as detector:
            frame_timestamp_ms = int(timestamps[i])
            # hand_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
            detection_result = detector.detect_for_video(mp_image, frame_timestamp_ms)
            print("第%d帧图像的边框数据" % i)
            print(detection_result) # 所获得的origin_x、origin_y应该是边框的左下角坐标
            # print(detection_result.detections[1].categories)
            # print(detection_result.detections[1].bounding_box)
            detection_results.append(detection_result)

    for i in range(len(frames)):
        hui_frame = frames[i]
        for j in range(0, shibie_num):
            category = detection_results[i].detections[j].categories[0]
            name = category.category_name

            bounding_box = detection_results[i].detections[j].bounding_box
            origin_x = bounding_box.origin_x
            origin_y = bounding_box.origin_y
            width = bounding_box.width
            height = bounding_box.height

            draw_juxing_name(hui_frame, origin_x, origin_y, width, height, name, j)

            if vis:
                cv2.imshow('Hand Landmarks Detection', hui_frame)

            if cv2.waitKey(33) & 0xFF == ord('q'):
                break


# 普遍识别为frisbee, bowl,



    # 释放视频文件并关闭窗口
    cap.release()
    cv2.destroyAllWindows()
    # return hand_pose_x, hand_pose_y, idx

if __name__ == '__main__':
    anniu()
