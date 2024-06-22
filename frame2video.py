import cv2
import os

def frames_to_video(frame_folder, output_video, fps=30):
    # 获取帧列表并排序
    frames = [f for f in os.listdir(frame_folder) if f.endswith('.png')]
    frames.sort()

    # 读取第一帧以获取帧的宽度和高度
    first_frame = cv2.imread(os.path.join(frame_folder, frames[0]))
    height, width, layers = first_frame.shape

    # 初始化视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # codec for .mp4 files
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for frame in frames:
        frame_path = os.path.join(frame_folder, frame)
        img = cv2.imread(frame_path)
        video.write(img)

    video.release()
    print(f"Video saved to {output_video}")

frame_folder = 'my_demo_frame/'
output_video = 'my_demo.mp4'
frames_to_video(frame_folder, output_video)