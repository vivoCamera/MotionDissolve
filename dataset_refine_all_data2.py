import os
import numpy as np
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF
from pdb import set_trace as stx
import random
import os, sys, math, random, glob, cv2, h5py, logging, random
import utils
import torch.utils.data as data

from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader



def binary_events_to_voxel_grid(events, num_bins, width, height):
    """
    Build a voxel grid with bilinear interpolation in the time domain from a set of events.

    :param events: a [N x 4] NumPy array containing one event per row in the form: [timestamp, x, y, polarity]
    :param num_bins: number of bins in the temporal axis of the voxel grid
    :param width, height: dimensions of the voxel grid
    """

    assert(events.shape[1] == 4)
    assert(num_bins > 0)
    assert(width > 0)
    assert(height > 0)

    voxel_grid = np.zeros((num_bins, height, width), np.float32).ravel()

    # normalize the event timestamps so that they lie between 0 and num_bins
    last_stamp = events[-1, 0]
    first_stamp = events[0, 0]
    deltaT = last_stamp - first_stamp

    if deltaT == 0:
        deltaT = 1.0

    events[:, 0] = (num_bins - 1) * (events[:, 0] - first_stamp) / deltaT
    ts = events[:, 0]
    xs = events[:, 1].astype(np.int32)
    ys = events[:, 2].astype(np.int32)
    pols = events[:, 3]

    pols[pols == 0] = -1
    tis = ts.astype(np.int32)
    dts = ts - tis

    vals_left = pols * (1.0 - dts)

    vals_right = pols * dts


    valid_indices = tis < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width
              + tis[valid_indices] * width * height, vals_left[valid_indices])

    valid_indices = (tis + 1) < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width
              + (tis[valid_indices] + 1) * width * height, vals_right[valid_indices])

    voxel_grid = np.reshape(voxel_grid, (num_bins, height, width))

    return voxel_grid


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])

class DataLoaderTrain(Dataset):
    def __init__(self, rgb_dir, args):
        super(DataLoaderTrain, self).__init__()

        self.inp_filenames=[]
        self.args=args
        train_file_dirs=rgb_dir
        for file_item in train_file_dirs:
            print(file_item)
            with h5py.File(file_item, 'r') as f:
                blur_image_name_list = sorted(list(f['blur_images'].keys()))
                sharp_image_name_list = sorted(list(f['sharp_images'].keys()))
                event_image_name_list = sorted(list(f['event_frames'].keys()))
                img_num = len(blur_image_name_list)
                for i in range(img_num):
                    blur_img = np.asarray(f['blur_images'][blur_image_name_list[i]]).transpose([2, 0, 1])
                    sharp_img = np.asarray(f['sharp_images'][sharp_image_name_list[i]]).transpose([2, 0, 1])
                    event_frame = np.asarray(f['event_frames'][event_image_name_list[i]])

                    # input_img, input_event, target = utils.image_proess(blur_img, event_frame, sharp_img,
                    #                                                     self.args.TRAINING.TRAIN_PS, self.args)
                    data = (blur_img, event_frame, sharp_img)
                    self.inp_filenames.append(data)

        self.sizex       = len(self.inp_filenames)  # get the size of target


    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        data=self.inp_filenames[index_]
        input_img, input_event, target = utils.image_proess(data[0], data[1], data[2],
                                                            self.args.TRAINING.TRAIN_PS, self.args)
        data = (input_img, input_event, target)
        return data







class SubsetSequentialSampler(Sampler):

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)



def create_data_loader(data_set, opts, mode='train'):

    # samples = 1000 * 32
    total_samples = opts.train_iters * opts.OPTIM.BATCH_SIZE
    print(total_samples)

    # 32000/950 约为 32 也就是重复32次便可以看到完成samples
    print(len(data_set))
    num_epochs = int(math.ceil(float(total_samples) / len(data_set)))

    indices = np.random.permutation(len(data_set))
    indices = np.tile(indices, num_epochs)
    indices = indices[:total_samples]

    ### generate data sampler and loader
    sampler = SubsetSequentialSampler(indices)
    # 这里shuffle是false的原意是因为indices已经随机了，所以这里不需要在shuffle = true
    data_loader = DataLoader(dataset=data_set, num_workers=10,
                             batch_size=opts.OPTIM.BATCH_SIZE, sampler=sampler, pin_memory=True,shuffle=False,drop_last=False)

    return data_loader


class DataLoaderTrain_npz(data.Dataset):

    def __init__(self, rgb_dir, args):
        super(DataLoaderTrain_npz, self).__init__()
        self.args = args

        self.blur_img_path = os.path.join(rgb_dir, 'aps_raw_int10')
        self.event_img_path = os.path.join(rgb_dir, 'process/events_17')
        self.shapr_img_path = os.path.join(rgb_dir, 'gt_raw_int10')

        # inp_files_dirs = sorted(os.listdir(self.blur_img_path))[:950]
        inp_files_dirs = sorted(os.listdir(self.blur_img_path))
        self.sequences_list = inp_files_dirs

        logging.info('[%s] Total %d event sequences:' %
                     (self.__class__.__name__, len(self.sequences_list)))
        print(len(self.sequences_list))

    def __len__(self):
        return len(self.sequences_list)

    def __getitem__(self, index):
        file_item = self.sequences_list[index]
        file_item_flag = file_item.split('.bin')[0]

        # 读取 blur 和 sharp 图像 (720x1280)
        blur_img = np.fromfile(os.path.join(self.blur_img_path, file_item), np.uint16).reshape(720, 1280).astype(np.float32) / 1023
        sharp_img = np.fromfile(os.path.join(self.shapr_img_path, file_item), np.uint16).reshape(720, 1280).astype(np.float32) / 1023

        blur_img = np.expand_dims(blur_img, axis=0)    # [1, 720, 1280]
        sharp_img = np.expand_dims(sharp_img, axis=0)  # [1, 720, 1280]

        # 读取 event frame (8, 360, 640)
        event = np.load(os.path.join(self.event_img_path, file_item_flag + '.npz'))
        event_frame = np.float32(event['evs'])  # [8, 360, 640] # event_frame.max:3.0 event_frame.min:-3.0

        # 进入 patch 切割
        blur_img, event_frame, sharp_img = utils.image_proess(
            blur_img, event_frame, sharp_img,
            self.args.TRAINING.TRAIN_PS, self.args
        )

        data = (blur_img, event_frame, sharp_img)
        return data


class DataLoaderTrain_npz_sub(data.Dataset):
    def __init__(self, rgb_dir, args):
        super(DataLoaderTrain_npz_sub, self).__init__()
        self.args = args
        self.current_epoch = 0  # 用于动态采样

        # 主路径
        self.blur_img_path_1 = os.path.join(rgb_dir, 'aps_raw_int10')
        self.event_img_path_1 = os.path.join(rgb_dir, 'process/events_17')
        self.sharp_img_path_1 = os.path.join(rgb_dir, 'gt_raw_int10')

        # 辅路径
        sub_rgb_dir = './GoPro/train/process'
        self.blur_img_path_2 = os.path.join(sub_rgb_dir, 'blur')
        self.event_img_path_2 = os.path.join(sub_rgb_dir, 'voxel_17_d2')
        self.sharp_img_path_2 = os.path.join(sub_rgb_dir, 'gt')

        # 主路径：随机采样 475 个 .bin 文件
        all_files_1 = sorted(os.listdir(self.blur_img_path_1))
        if len(all_files_1) < 475:
            raise ValueError("主路径图像不足 475")
        sampled_files_1 = random.sample(all_files_1, 475)
        self.data_list_1 = [(f, 1) for f in sampled_files_1]

        # 子路径：随机采样 475 个 .npz 文件
        all_files_2 = sorted(os.listdir(self.blur_img_path_2))
        if len(all_files_2) < 475:
            raise ValueError("子路径图像不足 475")
        sampled_files_2 = random.sample(all_files_2, 475)
        self.data_list_2 = [(f, 2) for f in sampled_files_2]

        self.total_len = 475 + 475
        logging.info('[%s] Total %d event sequences:' % (self.__class__.__name__, self.total_len))
        print(self.total_len)

    def __len__(self):
        return self.total_len

    def set_epoch(self, epoch):
        """外部调用，用于调整采样策略"""
        self.current_epoch = epoch

    def __getitem__(self, index):
        # 动态采样策略，线性增加子路径采样概率
        # milestones = [20, 70, 120, 170, 220, 270, 320, 370, 400]
        # max_p_sub = 0.5  # 最多 475/950
        
        milestones = [30, 60, 90, 120, 150]
        max_p_sub = 0.3  # 最多 475/950

        if self.current_epoch < milestones[0]:
            p_sub = 0.0
        elif self.current_epoch >= milestones[-1]:
            p_sub = max_p_sub
        else:
            for i in range(1, len(milestones)):
                if self.current_epoch < milestones[i]:
                    start_epoch = milestones[i - 1]
                    end_epoch = milestones[i]
                    alpha = (self.current_epoch - start_epoch) / (end_epoch - start_epoch)
                    p_sub = max_p_sub * ((i - 1 + alpha) / (len(milestones) - 1))
                    break

        use_sub_path = (random.random() < p_sub)

        if use_sub_path:
            file_item, path_flag = random.choice(self.data_list_2)
        else:
            file_item, path_flag = random.choice(self.data_list_1)

        if path_flag == 1:
            # 主路径 bin 文件
            file_item_flag = file_item.split('.bin')[0]

            blur_img = np.fromfile(os.path.join(self.blur_img_path_1, file_item), np.uint16).reshape(720, 1280).astype(np.float32) / 1023
            sharp_img = np.fromfile(os.path.join(self.sharp_img_path_1, file_item), np.uint16).reshape(720, 1280).astype(np.float32) / 1023

            blur_img = np.expand_dims(blur_img, axis=0)
            sharp_img = np.expand_dims(sharp_img, axis=0)

            event = np.load(os.path.join(self.event_img_path_1, file_item_flag + '.npz'))
            event_frame = np.float32(event['evs'])
            # event_frame = event_frame / np.max(event_frame)

        else:
            # 辅路径 npz 文件
            file_item_flag = os.path.splitext(file_item)[0]

            blur_npz = np.load(os.path.join(self.blur_img_path_2, file_item))
            sharp_npz = np.load(os.path.join(self.sharp_img_path_2, file_item))
            event_npz = np.load(os.path.join(self.event_img_path_2, file_item))

            blur_img = np.expand_dims(blur_npz['bin'], axis=0)
            sharp_img = np.expand_dims(sharp_npz['bin'], axis=0)
            event_frame = event_npz['arr_0']
            event_frame = event_frame.transpose(2, 0, 1)
            # event_frame = event_frame / np.max(event_frame)

        # patch 切割
        blur_img, event_frame, sharp_img = utils.image_proess(
            blur_img, event_frame, sharp_img,
            self.args.TRAINING.TRAIN_PS, self.args
        )

        return blur_img, event_frame, sharp_img


class DataLoaderVal_npz(Dataset):
    def __init__(self,rgb_dir, args):
        # super(DataLoaderTrain, self).__init__()

        # TODO 训练的时候从train里面划分50个出来
        # rgb_dir ./mipi2025/Train/process
        self.rgb_dir=rgb_dir
        self.blur_img_path = os.path.join(rgb_dir, 'aps_raw_int10')
        self.event_img_path = os.path.join(rgb_dir, 'process/events_17')
        self.shapr_img_path = os.path.join(rgb_dir, 'gt_raw_int10')

        inp_files_dirs = sorted(os.listdir(self.blur_img_path))[950:]

        self.DVS_stream_height = 720
        self.DVS_stream_width = 1280

        self.args=args

        self.sequences_list = inp_files_dirs
        # 选取该数据集下的后50个为valid

    def __len__(self):
        return len(self.sequences_list)
    
    def __getitem__(self, index):
        file_item = self.sequences_list[index] # 0382.bin
        file_item_flag = file_item.split('.bin')[0]
        # print(file_item)
        
        blur_img = np.fromfile(os.path.join(self.blur_img_path, file_item), np.uint16).reshape(720, 1280).astype(np.float32) / 1023

        sharp_img = np.fromfile(os.path.join(self.shapr_img_path, file_item), np.uint16).reshape(720, 1280).astype(np.float32) / 1023

        event = np.load(os.path.join(self.event_img_path, (file_item_flag + '.npz')))

        event_frame = np.float32(event['evs'])

        blur_img = np.expand_dims(blur_img, axis=0)
        sharp_img = np.expand_dims(sharp_img, axis=0)
        # print('blur_img.shape:{}'.format(blur_img.shape)) # (1, 720, 1280)
        # print('sharp_img.shape:{}'.format(sharp_img.shape)) # (1, 720, 1280)

        # event_frame = event_frame / np.max(event_frame)


        blur_img = torch.from_numpy(blur_img)
        sharp_img = torch.from_numpy(sharp_img)
        event_frame = torch.from_numpy(event_frame)
        
        data=(blur_img,event_frame,sharp_img)

        return data