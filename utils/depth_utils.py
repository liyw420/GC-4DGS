import torch
from utils.DV2.metric_depth.depth_anything_v2.dpt import DepthAnythingV2

# midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# midas.to(device)
# midas.eval()
# for param in midas.parameters():
#     param.requires_grad = False

# midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
# transform = midas_transforms.dpt_transform
# downsampling = 1

model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

depth_anything = DepthAnythingV2(**{**model_configs['vitl'], 'max_depth': 26.46764})
depth_anything.load_state_dict(torch.load('/media/vincent/HDD-02/fs4dgs/utils/DV2/metric_depth/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth', map_location='cpu'))
depth_anything = depth_anything.to('cuda').eval()

def estimate_depth_MiDas(img, mode='test'):
    h, w = img.shape[1:3]
    norm_img = (img[None] - 0.5) / 0.5
    norm_img = torch.nn.functional.interpolate(
        norm_img,
        size=(384, 512),
        mode="bicubic",
        align_corners=False)

    if mode == 'test':
        with torch.no_grad():
            prediction = midas(norm_img)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(h//downsampling, w//downsampling),
                mode="bicubic",
                align_corners=False,
            ).squeeze()
    else:
        prediction = midas(norm_img)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(h//downsampling, w//downsampling),
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    return prediction

def estimate_depth_DV2(rendered_image, input_size = 600):
    
    rendered_image = torch.clamp(rendered_image.permute(1, 2, 0), 0, 1) * 255
    rendered_image = rendered_image.cpu().detach().numpy()
    depth = depth_anything.infer_image(rendered_image, input_size)
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    depth = torch.tensor(depth).to('cuda')
    return depth