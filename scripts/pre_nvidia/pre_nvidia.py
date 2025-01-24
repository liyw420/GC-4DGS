import os
import logging
from argparse import ArgumentParser
import shutil
from PIL import Image

# 缩放并保存图片的函数
def resize_and_save_image(input_path, output_path):
    with Image.open(input_path) as img:
        # 将图片缩放为原来的1/2
        img_resized = img.resize((img.width // 2, img.height // 2))
        # 保存缩放后的图片到输出路径
        img_resized.save(output_path)


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

frame_gap = {'balloon1': 2900, 'balloon2': 3300, 'skating': 1100}

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

# # Resize images to half resolution
# for subdir, dirs, files in os.walk(args.source_path):
#     for file in files:
#         if file.endswith('.jpg'):
#             # 从文件路径中提取cam_name和frame
#             parts = subdir.split(os.sep)
#             cam_name = file.split('.')[0]
#             frame = int(parts[-2]) - int(frame_gap[parts[-4]])
            
#             # 构建输入和输出路径
#             input_path = os.path.join(subdir, file)
#             output_subdir = os.path.join(args.source_path, 'images_half')
#             os.makedirs(output_subdir, exist_ok=True)
#             output_path = os.path.join(output_subdir, f"{cam_name}_{frame:04d}.png")
            
#             # 缩放并保存图片
#             resize_and_save_image(input_path, output_path)

# print("所有图片已成功缩放并保存。")

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
