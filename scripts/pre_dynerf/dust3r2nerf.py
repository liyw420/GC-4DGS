from collections import OrderedDict
import os
import argparse
import glob
import numpy as np
import json
import sys
import math
import shutil
import sqlite3

def do_system(arg):
	print(f"==== running: {arg}")
	err = os.system(arg)
	if err:
		print("FATAL: command failed")
		sys.exit(err)

if __name__ == '__main__':
	parser = argparse.ArgumentParser() 
	parser.add_argument("--path", default="", help="input path to the video")
	args = parser.parse_args()

	# path must end with / to make sure image path is relative
	if args.path[-1] != '/':
		args.path += '/'
		
	# # extract images
	# videos = [os.path.join(args.path, vname) for vname in os.listdir(args.path) if vname.endswith(".mp4")]
	# images_path = os.path.join(args.path, "images/")
	# os.makedirs(images_path, exist_ok=True)
	
	# for video in videos:
	# 	cam_name = video.split('/')[-1].split('.')[-2]
	# 	do_system(f"ffmpeg -i {video} -vf scale=iw/2:ih/2 -t 10 -start_number 0 {images_path}/{cam_name}_%04d.png")  # 提取视频中的图片
	# # 为了降低def3dgs算法内存，将图片缩放为原来的1/2, 对应命令为 -vf scale=iw/2:ih/2
	
	# load image data
	images = [f[len(args.path):] for f in sorted(glob.glob(os.path.join(args.path, "images/", "*"))) if f.lower().endswith('png') or f.lower().endswith('jpg') or f.lower().endswith('jpeg')]
	cams = sorted(set([im[7:12] for im in images]))
	N = len(cams)

	# load camera intrinsics and extrinsics
	scale = 2704 / 512 / 2
	outdir = args.path
	intrinsics = np.load(os.path.join(outdir, 'intrinsics.npy'))
	focals = np.load(os.path.join(outdir, 'focals.npy'))
	extrinsics = np.load(os.path.join(outdir, 'extrinsics.npy'))
	bounds = np.load(os.path.join(outdir, 'bounds.npy'))
	blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
	
	# scale and transform the intrinsics and extrinsics
	fl = (focals * scale).tolist()
	fl = np.mean(np.array(fl).flatten()).tolist()
	W = int(intrinsics[0, :, 2][0].item() * scale * 2)
	H = int(intrinsics[0, :, 2][1].item() * scale * 2)

	# write all parameters to a json file
	train_frames = []
	test_frames = []
	for i in range(N):
		cam_frames = [{'file_path': im.lstrip("/").split('.')[0], 
					'transform_matrix': (extrinsics[i] @ np.linalg.inv(blender2opencv)).tolist(),
					'time': int(im.lstrip("/").split('.')[0][-4:]) / 30,                 # 调整每帧图片对应的时间戳
					'bounds': bounds[i].tolist()} for im in images if cams[i] in im]     # 添加一组参数bounds,为图像的最大最小深度值，用于计算相机的视锥体
		if i == 0:
			test_frames += cam_frames
		else:
			train_frames += cam_frames

	train_transforms = {
		'w': W,
		'h': H,
		'fl_x': fl,
		'fl_y': fl, 
		'cx': W // 2,
		'cy': H // 2,
		'frames': train_frames,
	}
	test_transforms = {
		'w': W,
		'h': H,
		'fl_x': fl,
		'fl_y': fl,
		'cx': W // 2,
		'cy': H // 2,
		'frames': test_frames,
	}

	train_output_path = os.path.join(args.path, 'transforms_train.json')
	test_output_path = os.path.join(args.path, 'transforms_test.json')
	print(f'[INFO] write to {train_output_path} and {test_output_path}')
	with open(train_output_path, 'w') as f:
		json.dump(train_transforms, f, indent=2)
	with open(test_output_path, 'w') as f:
		json.dump(test_transforms, f, indent=2)
	