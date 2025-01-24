import os
from PIL import Image
import torch
from torchvision import transforms

# 定义文件夹路径
input_folder = '/home/vincent/Pictures/figures'

# 用于将PIL图像转换为张量的变换
transform = transforms.ToTensor()
# 用于将张量转换为PIL图像的逆变换
to_pil = transforms.ToPILImage()

# 遍历文件夹内所有文件
for filename in os.listdir(input_folder):
    if filename.endswith('.png') and 'mask' not in filename:
        # 构造文件名
        base_name = filename.replace('.png', '')
        mask_filename = f"{base_name}_mask.png"
        output_filename = f"{base_name}_re.png"
        
        # 确保对应的mask文件存在
        if not os.path.exists(os.path.join(input_folder, mask_filename)):
            print(f"Mask file for {filename} not found, skipping.")
            continue

        # 读取图像和mask为PIL格式
        img = Image.open(os.path.join(input_folder, filename)).convert('RGB')
        mask = Image.open(os.path.join(input_folder, mask_filename)).convert('RGB')

        # 转换为张量
        img_tensor = transform(img)
        mask_tensor = transform(mask)
        
        # 执行哈达马积
        result_tensor = img_tensor * mask_tensor
        
        # 找到所有值为0的位置，然后将这些位置设置为白色
        zero_mask = result_tensor == 0
        white_tensor = torch.tensor([1.0, 1.0, 1.0]).view(3, 1, 1)  # RGB白色
        result_tensor = torch.where(zero_mask, white_tensor.expand_as(result_tensor), result_tensor)
        
        # 转换回PIL图像格式
        result_img = to_pil(result_tensor)
        
        # 保存结果图像
        result_img.save(os.path.join(input_folder, output_filename))
        print(f"Saved: {output_filename}")