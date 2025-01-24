
import cv2
import glob
import os

def create_video_from_images(image_folder, output_video_path, fps=30):
    # 获取所有 PNG 文件的路径
    image_files = sorted(glob.glob(os.path.join(image_folder, "*.png")))

    # 检查是否有 PNG 文件
    if not image_files:
        print("No PNG files found in the specified directory.")
        return

    # 读取第一张图片以获取视频帧的尺寸
    first_image = cv2.imread(image_files[0])
    height, width, layers = first_image.shape

    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4v 编码器
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # 逐帧写入视频
    for image_file in image_files:
        img = cv2.imread(image_file)
        video.write(img)

    # 释放视频写入对象
    video.release()
    print(f"Video saved to {output_video_path}")

image_folder = "/media/vincent/HDD-02/E-D3DGS/output/technicolor/Birthday/test/ours_60000/renders"
output_video_path = "/media/vincent/HDD-02/E-D3DGS/output/technicolor/Birthday/test/ours_60000/renders/output_video.mp4"
create_video_from_images(image_folder, output_video_path)