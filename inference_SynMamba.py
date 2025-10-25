import argparse
import cv2
import glob
import os
from tqdm import tqdm
import torch
from yaml import load

from basicsr.utils import img2tensor, tensor2img, imwrite
from basicsr.archs.synmamba_arch import SynMamba
from basicsr.utils.download_util import load_file_from_url
import numpy as np
import torch

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
from skimage.metrics import structural_similarity as ssim
import pyiqa
import torch.nn.functional as F
import os
import cv2
import numpy as np
import torch
import lpips
from PIL import Image
import math
from torchvision.utils import save_image 


loss_fn_alex = lpips.LPIPS(net='alex').cuda()

# 全局只创建一次度量器，避免重复加载
psnr_metric = pyiqa.create_metric('psnr', test_y_channel=False, color_space='rgb')
ssim_metric = pyiqa.create_metric('ssim', test_y_channel=False, color_space='rgb')
def calculate_psnr(img1, img2):
    """
    img1, img2: H×W×3  float32 0-255 或 uint8
    返回 float，越高越好
    """
    # 转 torch: [1,3,H,W]  0-1 float
    img1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0) / 255.0
    img2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0) / 255.0
    return psnr_metric(img1, img2).item()


def calculate_ssim(img1, img2):
    """
    img1, img2: H×W×3  float32 0-255 或 uint8
    返回 float 0-1，越高越好
    """
    img1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0) / 255.0
    img2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0) / 255.0
    return ssim_metric(img1, img2).item()
def print_network(model):
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(model)
    print("The number of parameters: {} M".format(float(num_params)/1000000.0))

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', type=str,
                        default='/root/shared-nvme/lowlight_data/LOLv2/LOLv2/Real_captured/Test/input',
                        help='Input image or folder')

    parser.add_argument('-g', '--gt', type=str,
                        default='/root/shared-nvme/lowlight_data/LOLv2/LOLv2/Real_captured/Test/gt',
                        help='groundtruth image')

    parser.add_argument('-o', '--output', type=str, default='results/LoLv2_Real', help='Output folder')
    parser.add_argument('-w', '--weight', type=str,
                        default='your .pth path',
                        help='path for model weights')

    parser.add_argument('-s', '--out_scale', type=int, default=1, help='The final upsampling scale of the image')
    parser.add_argument('--suffix', type=str, default='', help='Suffix of the restored image')
    parser.add_argument('--max_size', type=int, default=600,help='Max image size for whole image inference, otherwise use tiled_test')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    enhance_weight_path = args.weight

    EnhanceNet = SynMamba(in_chn=3,
             wf=24,
             n_blocks=[1,1,1,1],
             ffn_scale=4,
             high_heads=[1,2,4],
             high_blocks=[2,2,4]).to(device)
                 
    EnhanceNet.load_state_dict(torch.load(enhance_weight_path)['params'], strict=False)
    EnhanceNet.eval()
    print_network(EnhanceNet)

    os.makedirs(args.output, exist_ok=True)

    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        paths = sorted(glob.glob(os.path.join(args.input, '*')))

    pbar = tqdm(total=len(paths), unit='image')

    lpips_list = []
    ssim_scores = []
    psnr_scores = []
    
    for idx, path in enumerate(paths):
        img_name = os.path.basename(path) 
        pbar.set_description(f'Test {img_name}')

        gt_path = args.gt 
        file_name = path.split('/')[-1]

        gt_img = cv2.imread(os.path.join(gt_path, file_name), cv2.IMREAD_UNCHANGED) 
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED) 

        
        img_tensor = img2tensor(img).to(device) / 255. 
        img_tensor = img_tensor.unsqueeze(0) 

        # ---------- 1. 前向推理 ----------
        EnhanceNet.eval()
        with torch.no_grad():
            output = EnhanceNet.test(img_tensor)        # [1,3,H,W]  0-1 float32
            save_path = os.path.join(args.output, f'{img_name}')
            save_image(output, save_path)
            output = torch.clamp(output, 0, 1)

        # ---------- 2. 构造张量 ----------
        out_np = tensor2img(output)                     # BGR uint8 0-255
        out_np = cv2.cvtColor(out_np, cv2.COLOR_BGR2RGB)
        out_np = out_np.astype(np.float32)              # 0-255 float32

        gt_np = cv2.imread(os.path.join(gt_path, file_name), cv2.IMREAD_UNCHANGED)
        gt_np = cv2.cvtColor(gt_np, cv2.COLOR_BGR2RGB).astype(np.float32)

        # ---------- 3. 计算 PSNR / SSIM ----------
        psnr_score = calculate_psnr(out_np, gt_np)
        ssim_score = calculate_ssim(out_np, gt_np)

        # ---------- 4. 计算 LPIPS ----------
        img_tensor_lpips = torch.from_numpy(out_np).permute(2, 0, 1).unsqueeze(0).cuda()
        gt_tensor_lpips  = torch.from_numpy(gt_np).permute(2, 0, 1).unsqueeze(0).cuda()
        lpips_alex = loss_fn_alex(img_tensor_lpips, gt_tensor_lpips).item()

        # ---------- 4. 记录 & 打印 ----------
        psnr_scores.append(psnr_score)
        ssim_scores.append(ssim_score)
        lpips_list.append(lpips_alex)

        print(f"{img_name}  PSNR={psnr_score:.4f}  SSIM={ssim_score:.4f}  LPIPS={lpips_alex:.4f}")


        pbar.update(1)
    pbar.close()
    print("Average SSIM: {}".format(np.mean(ssim_scores)))
    print("Average PSNR: {}".format(np.mean(psnr_scores)))
    print("Average LPIPS: {}".format(np.mean(lpips_list)))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # EnhanceNet = SynMamba(in_chn=3,
    #          wf=24,
    #          n_blocks=[1,1,1],
    #          ffn_scale=4,
    #          high_heads=[2,4,8],
    #          high_blocks=[4,4,4]).to(device)
    # print_network(EnhanceNet)
    main()
