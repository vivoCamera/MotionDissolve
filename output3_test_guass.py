import torch
import os
from config import Config
import cv2
import torch.nn as nn
import random
import time
import numpy as np
import utils


from U_model import unet2_trans_3d_fus3

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

    model_restoration = unet2_trans_3d_fus3.Restoration(1, 17, 1, opt)
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
        # event_frame = event_frame.transpose([1, 2, 0])
        # new_height = event_frame.shape[0] * 2
        # new_width = event_frame.shape[1] * 2
        # upsampled_event_frame = np.zeros((new_height, new_width, event_frame.shape[2]), dtype=event_frame.dtype)
        # for i in range(event_frame.shape[2]):
        #     upsampled_event_frame[:, :, i] = cv2.resize(event_frame[:, :, i], (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        # event_frame = upsampled_event_frame.transpose([2, 0, 1])
       
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

    # times = []
    # patch_size = 256
    # overlap = 128
    # stride = patch_size - overlap
    # gauss_mask = get_gaussian_weight(patch_size, patch_size).cuda()

    # for idx, val_file in enumerate(inp_files_dirs):
    #     item = val_file.split('/')[-1].split('.bin')[0]
    #     blur_img_np = np.fromfile(os.path.join(input_dir, (item + '.bin')), np.uint16).reshape(720, 1280).astype(np.float32) / 1023
    #     event = np.load(os.path.join(event_dir, (item + '.npz')))
    #     event_frame_np = np.float32(event['evs'])
    #     event_frame_np = event_frame_np / event_frame_np.max()
        
        
    #     H, W = blur_img_np.shape
    #     H_e, W_e = event_frame_np.shape[1:]

    #     assert H_e * 2 == H and W_e * 2 == W, f"event_frame size {H_e}x{W_e} is not half of blur_img {H}x{W}"

    #     output_img = torch.zeros((1, 1, H, W), dtype=torch.float32).cuda()
    #     norm_mask = torch.zeros((1, 1, H, W), dtype=torch.float32).cuda()

    #     torch.cuda.synchronize()
    #     start_time = time.time()

    #     with torch.no_grad():
    #         for top in range(0, H, stride):
    #             for left in range(0, W, stride):
    #                 bottom = min(top + patch_size, H)
    #                 right = min(left + patch_size, W)
    #                 top = max(0, bottom - patch_size)
    #                 left = max(0, right - patch_size)

    #                 patch_blur = blur_img_np[top:bottom, left:right]
    #                 etop = top // 2
    #                 ebottom = bottom // 2
    #                 eleft = left // 2
    #                 eright = right // 2
    #                 patch_event = event_frame_np[:, etop:ebottom, eleft:eright]

    #                 patch_blur = torch.from_numpy(patch_blur).unsqueeze(0).unsqueeze(0).cuda()
    #                 patch_event = torch.from_numpy(patch_event).unsqueeze(0).cuda()

    #                 restored_patch = model_restoration(patch_blur, patch_event)

    #                 # 裁剪高斯权重以匹配当前 patch 尺寸
    #                 gw = gauss_mask[:, :, :bottom-top, :right-left]
    #                 output_img[:, :, top:bottom, left:right] += restored_patch * gw
    #                 norm_mask[:, :, top:bottom, left:right] += gw

    #     torch.cuda.synchronize()
    #     end_time = time.time()
    #     inference_time = end_time - start_time
    #     times.append(inference_time)
    #     print(f"Image {item}: {inference_time:.4f} seconds")

    #     output_img = output_img / norm_mask
    #     output_img = (output_img * 1023).clamp_(0.0, 1023.0).round().squeeze().cpu().numpy().astype(np.uint16)
    #     output_img.tofile(os.path.join(bin_out_dir, (item + '.bin')))

    # avg_time = sum(times) / len(times)
    # print(f"\nAverage inference time per image: {avg_time:.4f} seconds")

if __name__ == '__main__':
    main()
