import cv2
import os


def split_video_to_frames(video_path, output_dir, frame_rate=30):
    # 读取视频文件
    video_capture = cv2.VideoCapture(video_path)

    # Get the total number of frames and the frame rate
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT)) # 视频文件的总帧数
    fps = int(video_capture.get(cv2.CAP_PROP_FPS)) # 视频帧率

    # Calculate the frame interval to capture one frame every `frame_rate` frames
    frame_interval = int(round(fps / frame_rate)) # 将视频的帧率除以所需的帧率（即frame_rate），然后四舍五入取整。这样可以得到一个适当的帧间隔，以便从视频中提取所需帧数的图像。

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True) # 判断输出目录是否存在，不存在则创建目录

    # Initialize frame counter
    frame_count = 0

    # Read and save frames at the specified interval
    while True:
        ret, frame = video_capture.read() # 读取帧，ret为是否读取成功的标志，frame为读取的帧
        if not ret:
            break
        if frame_count % frame_interval == 0: # 判断是否该保存该帧
            # frame_name = f"frame_{frame_count // frame_interval}.jpg"
            frame_name = f"{frame_count // frame_interval}.jpg"
            frame_path = os.path.join(output_dir, frame_name)
            cv2.imwrite(frame_path, frame)
        frame_count += 1
    frame_time = 1/frame_rate
    # print(frame_time)
    print("总共%d帧图片已输出" % frame_count)
    print("一帧%fs" % frame_time)

    # Release the video capture object
    video_capture.release() # 释放内存
    return frame_time


# # Example usage:
# video_path = "D:/hand_data/test_video/test_1.mp4"
# output_dir = "./image/"
# split_video_to_frames(video_path, output_dir)
