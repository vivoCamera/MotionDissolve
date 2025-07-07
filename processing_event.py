import os
import numpy as np
from omegaconf import OmegaConf
from colour_demosaicing import demosaicing_CFA_Bayer_Menon2007
import torch
import matplotlib.pyplot as plt
import os
from PIL import Image
import torch.nn.functional as F

# 函数：使用Menon2007算法对Bayer图像进行去马赛克
def demosaic_Menon2007(bayer, cfa='RGGB'):
    # 取出bayer图像的第一个batch、第一个通道，转为numpy数组
    bayer_np = bayer[0, 0].cpu().numpy()
    # 使用Menon2007算法进行去马赛克
    rgb_np = demosaicing_CFA_Bayer_Menon2007(bayer_np, pattern=cfa)
    # 转为FloatTensor，并调整维度顺序为 (1, 3, H, W)
    rgb_pt = torch.FloatTensor(rgb_np).permute(2, 0, 1).unsqueeze(0).to(bayer.device)
    return rgb_pt


# 函数：应用白平衡增益（AWB）
def apply_awb(raw_or_rgb, awb_gain):
    awb_gain = awb_gain.unsqueeze(1)  # 调整维度 (1, 1, 3)
    out = torch.zeros_like(raw_or_rgb)  # 初始化输出
    # 分别对四个马赛克通道应用不同的增益
    out[:, :, ::2, ::2] = raw_or_rgb[:, :, ::2, ::2] * awb_gain[:, :, :, 0]  # R通道
    out[:, :, 1::2, 1::2] = raw_or_rgb[:, :, 1::2, 1::2] * awb_gain[:, :, :, 2]  # B通道
    out[:, :, ::2, 1::2] = raw_or_rgb[:, :, ::2, 1::2] * awb_gain[:, :, :, 1]  # G通道
    out[:, :, 1::2, ::2] = raw_or_rgb[:, :, 1::2, ::2] * awb_gain[:, :, :, 1]  # G通道
    out = out.clamp(0., 1.)  # 限幅在[0,1]之间
    return out


# 函数：应用颜色校正矩阵（CCM）
def apply_ccm(image, cam2rgbs):
    image = image.permute(0, 2, 3, 1)  # 变为 (B, H, W, C)
    shape = image.shape
    image = image.contiguous().view(-1, 3, 1)  # 展平像素为 (N, 3, 1)

    cam2rgbs = cam2rgbs.expand(-1, shape[1]*shape[2], -1, -1)  # 扩展到每个像素
    cam2rgbs = cam2rgbs.contiguous().view(-1, 3, 3)  # 变为 (N, 3, 3)

    image = torch.bmm(cam2rgbs, image)  # 颜色校正
    image = image.squeeze(-1).view(shape)  # 恢复为 (B, H, W, C)
    image = image.permute(0, 3, 1, 2)  # 变为 (B, C, H, W)
    image = image.clamp(0., 1.)  # 限幅
    return image


# 函数：ISP处理流程（白平衡、去马赛克、颜色校正）
def process_isp(raw, conf):
    # 从配置文件中读取AWB增益和颜色校正矩阵
    awb_gain = torch.FloatTensor(conf.simulator.meta.awb_gain).view(1, 1, 3)
    cam2rgb = torch.FloatTensor(conf.simulator.meta.cam2rgb).view(1, 1, 3, 3)
    # 归一化raw图像到0-1之间
    raw = torch.FloatTensor(raw.astype(np.float32)).unsqueeze(0).unsqueeze(0) / 1023
    raw = apply_awb(raw, awb_gain)  # 应用白平衡
    rgb = demosaic_Menon2007(raw.clamp(1e-4, 1))  # 去马赛克
    rgb = apply_ccm(rgb, cam2rgb)  # 应用颜色校正
    rgb1 = np.power(np.clip(rgb, 0, 1), 1/2.2)  # Gamma矫正
    rgb1 = (rgb1 * 255.).clamp(0., 255.)  # 转为0-255范围
    return rgb1

def read_aps(dp):
    aps = np.fromfile(dp, np.uint16).reshape(720, 1280)  
    return aps

def decompress_from_2bit(compressed):
    unpacked = np.zeros(compressed.size * 4, dtype=np.uint8) 
    unpacked[::4] = (compressed >> 6) & 0b11  
    unpacked[1::4] = (compressed >> 4) & 0b11  
    unpacked[2::4] = (compressed >> 2) & 0b11  
    unpacked[3::4] = compressed & 0b11  
    reverse_mapping = {0: -1, 1: 0, 2: 1}
    decoded = np.vectorize(reverse_mapping.get)(unpacked)
    return decoded.reshape((1, -1, 360, 640))  # (B, T, H, W)

def read_evs(dp):
    evs_compress = np.fromfile(dp, dtype=np.int8)  
    evs = decompress_from_2bit(evs_compress)  
    return evs


# 函数：将变帧率的事件帧合并为固定帧率
def binning_frame(evs_frame, out_frame=8):
    b, c, h, w = evs_frame.shape
    out_evs = torch.zeros((b, out_frame, h, w))  # 初始化输出
    for i in range(out_frame):
        start_num = c / out_frame * i  # 当前输出帧的起始时间戳
        end_num = c / out_frame * (i+1)  # 当前输出帧的结束时间戳
        k_start = int(np.floor(start_num))
        k_end = int(np.ceil(end_num))
        for k in range(k_start, k_end):
            if k < 0 or k >= c:
                continue  # 越界检查
            seq_start = max(start_num, k)
            seq_end = min(end_num, k+1)
            weights = seq_end - seq_start  # 计算权重
            out_evs[:, i, :, :] += evs_frame[:, k] * weights  # 累加加权帧
    return out_evs


def binning_frame_fixed2(evs_frame, out_frame=17):

    # 若是 numpy，转为 tensor
    if isinstance(evs_frame, np.ndarray):
        evs_frame = torch.from_numpy(evs_frame)

    if not isinstance(evs_frame, torch.Tensor):
        raise TypeError("evs_frame must be torch.Tensor or np.ndarray")

    evs_frame = evs_frame.float()  # 确保 float 类型
    B, T, H, W = evs_frame.shape

    if T == out_frame:
        return evs_frame

    elif T < out_frame:
        evs_reshaped = evs_frame.permute(0, 2, 3, 1).reshape(-1, 1, T)
        evs_interp = F.interpolate(evs_reshaped, size=out_frame, mode='linear', align_corners=True)
        evs_interp = evs_interp.reshape(B, H, W, out_frame).permute(0, 3, 1, 2)  # [B, out_frame, H, W]
        return evs_interp

    else:
        out_evs = torch.zeros((B, out_frame, H, W), device=evs_frame.device)
        for i in range(out_frame):
            start_num = T / out_frame * i
            end_num = T / out_frame * (i + 1)
            k_start = int(np.floor(start_num))
            k_end = int(np.ceil(end_num))
            for k in range(k_start, k_end):
                if k < 0 or k >= T:
                    continue
                seq_start = max(start_num, k)
                seq_end = min(end_num, k + 1)
                weight = seq_end - seq_start
                out_evs[:, i] += evs_frame[:, k] * weight
        return out_evs
    
if __name__ == '__main__':
    import glob


    evs_folder = './finetune_2stage/data/Test/events/'
    save_dir = './finetune_2stage/data/Test/process/events_17_2'
    os.makedirs(save_dir, exist_ok=True)

    bin_files = sorted(glob.glob(os.path.join(evs_folder, '*.bin')))
    print(f"Found {len(bin_files)} EVS bin files.")

    for idx, evs_path in enumerate(bin_files):
        evs = read_evs(evs_path)  # (B, T, H, W)
        # print(evs.shape)
        # evs_binned = binning_frame(evs, out_frame=11)  # (B, 24, H, W)
        print(evs.shape)
        evs_binned = binning_frame_fixed2(evs, out_frame=17)  # (B, 24, H, W)
        # print(evs_binned.shape)
        evs_binned_np = evs_binned[0].numpy()  # 转为numpy
        # print(evs_binned_np.shape)

        # 保存为npz
        basename = os.path.basename(evs_path).replace('.bin', '.npz')
        save_path = os.path.join(save_dir, basename)
        np.savez_compressed(save_path, evs=evs_binned_np)

        print(f"[{idx+1}/{len(bin_files)}] Saved EVS npz to {save_path}")




