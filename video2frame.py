import cv2
import os

def extract_frames(video_path, output_dir):
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 如果没有读取到帧，退出循环

        # 构建输出文件名
        output_file = os.path.join(output_dir, f"frame_{frame_index:04d}.png")

        # 保存当前帧为PNG文件
        cv2.imwrite(output_file, frame)
        frame_index += 1

    cap.release()
    print(f"Extracted {frame_index} frames and saved to '{output_dir}'.")

# 示例用法
video_path = 'video.mp4'
output_dir = 'my_demo/'
extract_frames(video_path, output_dir)