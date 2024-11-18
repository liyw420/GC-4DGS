
import os
import logging
from argparse import ArgumentParser
import shutil
from PIL import Image

def copy_and_rename_images(source_dir, dest_dir, num_frames=100):
    # 确保目标目录存在
    os.makedirs(dest_dir, exist_ok=True)
    
    # 遍历相机文件夹
    for cam_folder in os.listdir(source_dir):
        cam_path = os.path.join(source_dir, cam_folder)
        
        # 检查是否为目录
        if os.path.isdir(cam_path):
            # 获取相机编号
            try:
                cam_number = int(cam_folder)
            except ValueError:
                continue
            
            # 获取图片列表并排序
            images = sorted(os.listdir(cam_path))
            
            # 复制前100帧
            for i, image_name in enumerate(images[:num_frames]):
                # 构建源文件路径
                src_image_path = os.path.join(cam_path, image_name)
                
                # 获取图片编号（假设文件名是数字.jpg）
                try:
                    image_number = int(os.path.splitext(image_name)[0])
                except ValueError:
                    continue
                
                # 打开图片并转换格式和分辨率
                with Image.open(src_image_path) as img:
                    # 调整分辨率为原来的一半
                    new_size = (img.width // 2, img.height // 2)
                    img_resized = img.resize(new_size, Image.LANCZOS)
                    
                    # 构建目标文件名和路径
                    new_image_name = f"cam{cam_number:02d}_{image_number:04d}.png"
                    dest_image_path = os.path.join(dest_dir, new_image_name)
                    
                    # 保存为PNG格式
                    img_resized.save(dest_image_path, 'PNG')
                    print(f"Converted and copied {src_image_path} to {dest_image_path}")


# This Python script is based on the shell converter script provided in the MipNerF 360 repository.
parser = ArgumentParser("Colmap converter")
parser.add_argument("--no_gpu", action='store_true')
parser.add_argument("--skip_matching", action='store_true')
parser.add_argument("--source_path", "-s", required=True, type=str)
parser.add_argument("--camera", default="OPENCV", type=str)
parser.add_argument("--colmap_executable", default="", type=str)
parser.add_argument("--resize", action="store_true")
parser.add_argument("--magick_executable", default="", type=str)
args = parser.parse_args()

colmap_command = '"{}"'.format(args.colmap_executable) if len(args.colmap_executable) > 0 else "colmap"
use_gpu = 1 if not args.no_gpu else 0

if not args.skip_matching:
    os.makedirs(args.source_path + "/distorted/sparse", exist_ok=True)

    ## Feature extraction
    feat_extracton_cmd = colmap_command + " feature_extractor "\
        "--database_path " + args.source_path + "/distorted/database.db \
        --image_path " + args.source_path + "/input \
        --ImageReader.single_camera 1 \
        --ImageReader.camera_model " + "PINHOLE" 
    exit_code = os.system(feat_extracton_cmd)

    ## Feature matching
    feat_matching_cmd = colmap_command + " exhaustive_matcher \
        --database_path " + args.source_path + "/distorted/database.db \
        --SiftMatching.use_gpu " + str(use_gpu)
    exit_code = os.system(feat_matching_cmd)

    ### Bundle adjustment
    # The default Mapper tolerance is unnecessarily large,
    # decreasing it speeds up bundle adjustment steps.
    mapper_cmd = (colmap_command + " mapper \
        --database_path " + args.source_path + "/distorted/database.db \
        --image_path "  + args.source_path + "/input \
        --output_path "  + args.source_path + "/distorted/sparse")
    exit_code = os.system(mapper_cmd)

### Image undistortion
## We need to undistort our images into ideal pinhole intrinsics.
img_undist_cmd = (colmap_command + " image_undistorter \
    --image_path " + args.source_path + "/input \
    --input_path " + args.source_path + "/distorted/sparse/0 \
    --output_path " + args.source_path + "/input")
exit_code = os.system(img_undist_cmd)

# Copy each file from the source directory to the destination directory
files = os.listdir(args.source_path + "/input/sparse")
os.makedirs(args.source_path + "/sparse/0", exist_ok=True)
for file in files:
    if file == 'project.ini':
        continue
    else:
        source_file = os.path.join(args.source_path, "input/sparse", file)
        destination_file = os.path.join(args.source_path, "sparse", "0", file)
        shutil.move(source_file, destination_file)

# Resize images to half resolution
    source_path = os.path.join(args.source_path, "images")
    des_path = os.path.join(args.source_path, "images_half")
    copy_and_rename_images(source_path, des_path)

# Delete files
items_to_delete = [
    "distorted",
    "run-colmap-geometric.sh",
    "run-colmap-photometric.sh",
    "stereo"
]
for item in items_to_delete:
    item_path = os.path.join(args.source_path, item)
    if os.path.exists(item_path):
        if os.path.isfile(item_path):
            os.remove(item_path)
            print(f"Deleted file: {item_path}")
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)
            print(f"Deleted directory: {item_path}")
    else:
        print(f"Item not found: {item_path}")
