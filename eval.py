import os
import cv2
import numpy as np
import torch
import lpips
from PIL import Image
import math
os.environ["CUDA_VISIBLE_DEVICES"]="0"

loss_fn_alex = lpips.LPIPS(net='alex').cuda()

def calculate_psnr(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_ssim(img1, img2):
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:, :, i], img2[:, :, i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

if __name__ == '__main__':
    # img_path = r'D:/MyFile/paper/our_proposed/Wave-Mamba/66000/66000'
    img_path = r'D:/MyFile/paper/our_proposed/Wave-Mamba/Figures/lolv1'
    # gt_path = r'D:/MyFile/paper/dataset/LOLv2/LOLv2/Synthetic/Test/Normal'
    gt_path = r'D:/MyFile/paper/dataset/LOLdataset/eval15/gt'
    file_list = os.listdir(img_path)
    lpips_list = []
    ssim_scores = []
    psnr_scores = []
    for filename in file_list:
        print("eval data " + filename)

        new_extension = 'png'
        new_filename = os.path.splitext(filename)[0] + '.' + new_extension
        new_filename = "normal" + filename

        img = cv2.imread(os.path.join(img_path,filename))
        filename1=filename.replace('low','normal')
        gt = cv2.imread(os.path.join(gt_path,filename1))
        print(filename,filename1)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(np.float32(img))
        gt = torch.from_numpy(np.float32(gt))
        img = img.permute(2, 0, 1).unsqueeze(0).cuda()
        gt = gt.permute(2, 0, 1).unsqueeze(0).cuda()
        lpips_alex = loss_fn_alex(img, gt)
        print("lpips", lpips_alex)
        lpips_alex = lpips_alex.detach().cpu().numpy()
        lpips_list.append(lpips_alex)

        test_img = Image.open(os.path.join(img_path, filename))

        gt_img = Image.open(os.path.join(gt_path, filename1))
        test_img = np.array(test_img).astype(np.uint8)
        gt_img = np.array(gt_img).astype(np.uint8)
        # 计算SSIM
        ssim_score = calculate_ssim(test_img, gt_img)
        ssim_scores.append(ssim_score)
        print("ssim", ssim_score)
        # 计算PSNR
        psnr_score = calculate_psnr(test_img, gt_img)
        psnr_scores.append(psnr_score)
        print("psnr", psnr_score)

    print("Average SSIM: {}".format(np.mean(ssim_scores)))
    print("Average PSNR: {}".format(np.mean(psnr_scores)))
    print("Average LPIPS: {}".format(np.mean(lpips_list)))