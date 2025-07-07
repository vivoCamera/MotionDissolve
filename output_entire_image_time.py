import torch
import os
from config import Config
import cv2
import torch.nn as nn
import random
import time
import numpy as np
import utils


from MotionDissolve.MotionDissolve.U_model import motiondissolve

from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR

opt = Config('./MotionDissolve/training_r_all2.yml')
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
torch.backends.cudnn.benchmark = True

random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

def get_gaussian_weight(h, w, sigma_scale=1./6):
    center_h = (h - 1) / 2.0
    center_w = (w - 1) / 2.0
    sigma_h = h * sigma_scale
    sigma_w = w * sigma_scale
    y, x = np.ogrid[:h, :w]
    gaussian = np.exp(-(((x - center_w) ** 2) / (2 * sigma_w ** 2) + ((y - center_h) ** 2) / (2 * sigma_h ** 2)))
    gaussian = gaussian / gaussian.max()
    return torch.from_numpy(gaussian).float().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

def main():
    input_dir = './MotionDissolve/data/Test/aps_raw_int10'
    event_dir = './MotionDissolve/data/Test/events_17_2'
    model_pth = './MotionDissolve/pth/refine.pth'

    model_restoration = motiondissolve.Restoration(1, 17, 1, opt)
    model_restoration.cuda()
    checkpoint = torch.load(model_pth)
    model_restoration.load_state_dict(checkpoint['state_dict'])
    model_restoration.eval()

    inp_files_dirs = sorted(os.listdir(input_dir))

    # Warm-up
    warmup_file = inp_files_dirs[0]
    warmup_item = warmup_file.split('/')[-1].split('.bin')[0]
    blur_img = np.fromfile(os.path.join(input_dir, (warmup_item + '.bin')), np.uint16).reshape(720, 1280).astype(np.float32) / 1023
    event = np.load(os.path.join(event_dir, (warmup_item + '.npz')))
    event_frame = np.float32(event['evs'])

    blur_img = np.expand_dims(blur_img, axis=0)
    blur_img = np.expand_dims(blur_img, axis=0)
    event_frame = np.expand_dims(event_frame, axis=0)

    blur_img = torch.from_numpy(blur_img).cuda()
    event_frame = torch.from_numpy(event_frame).cuda()
    with torch.no_grad():
        for i in range(10):
            _ = model_restoration(blur_img, event_frame)

    print("Warm-up done!")

    times = []
    for idx, val_file in enumerate(inp_files_dirs):
        item = val_file.split('/')[-1].split('.bin')[0]
        blur_img = np.fromfile(os.path.join(input_dir, (item + '.bin')), np.uint16).reshape(720, 1280).astype(np.float32) / 1023
        print(blur_img.shape) # (720, 1280)

        event = np.load(os.path.join(event_dir, (item + '.npz')))
        event_frame = np.float32(event['evs'])
        
        event_frame = event_frame / np.max(event_frame)

        print(event_frame.shape)
       
        blur_img = np.expand_dims(blur_img, axis=0)
        blur_img = np.expand_dims(blur_img, axis=0)
        event_frame = np.expand_dims(event_frame, axis=0)

        blur_img = torch.from_numpy(blur_img).cuda()
        event_frame = torch.from_numpy(event_frame).cuda()

        torch.cuda.synchronize()  # 确保计时准确
        start_time = time.time()

        with torch.no_grad():
            restored = model_restoration(blur_img, event_frame)

        torch.cuda.synchronize()
        end_time = time.time()

        inference_time = end_time - start_time
        times.append(inference_time)
        print(f"Image {item}: {inference_time:.4f} seconds")

    avg_time = sum(times) / len(times)
    print(f"\nAverage inference time per image: {avg_time:.4f} seconds")

if __name__ == '__main__':
    main()
