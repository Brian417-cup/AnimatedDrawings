import os.path as osp
import sys

sys.path.append(osp.abspath(osp.join(osp.dirname(__file__))))
sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '..')))
sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '..', '..')))

import torch
import numpy as np
import glob
import os
import io
import math
import random
import json
import pickle
import math
from torch.utils.data import Dataset, DataLoader
from .utils_data import crop_scale
import imageio


def halpe2h36m(x):
    '''
        Input: x (T x V x C)  
       //Halpe 26 body keypoints
    {0,  "Nose"},
    {1,  "LEye"},
    {2,  "REye"},
    {3,  "LEar"},
    {4,  "REar"},
    {5,  "LShoulder"},
    {6,  "RShoulder"},
    {7,  "LElbow"},
    {8,  "RElbow"},
    {9,  "LWrist"},
    {10, "RWrist"},
    {11, "LHip"},
    {12, "RHip"},
    {13, "LKnee"},
    {14, "Rknee"},
    {15, "LAnkle"},
    {16, "RAnkle"},
    {17,  "Head"},
    {18,  "Neck"},
    {19,  "Hip"},
    {20, "LBigToe"},
    {21, "RBigToe"},
    {22, "LSmallToe"},
    {23, "RSmallToe"},
    {24, "LHeel"},
    {25, "RHeel"},
    '''
    T, V, C = x.shape
    y = np.zeros([T, 17, C])
    y[:, 0, :] = x[:, 19, :]
    y[:, 1, :] = x[:, 12, :]
    y[:, 2, :] = x[:, 14, :]
    y[:, 3, :] = x[:, 16, :]
    y[:, 4, :] = x[:, 11, :]
    y[:, 5, :] = x[:, 13, :]
    y[:, 6, :] = x[:, 15, :]
    y[:, 7, :] = (x[:, 18, :] + x[:, 19, :]) * 0.5
    y[:, 8, :] = x[:, 18, :]
    y[:, 9, :] = x[:, 0, :]
    y[:, 10, :] = x[:, 17, :]
    y[:, 11, :] = x[:, 5, :]
    y[:, 12, :] = x[:, 7, :]
    y[:, 13, :] = x[:, 9, :]
    y[:, 14, :] = x[:, 6, :]
    y[:, 15, :] = x[:, 8, :]
    y[:, 16, :] = x[:, 10, :]
    return y


# 将coco格式的16个关键点转换为H36M格式的关键点格式
def coco2h36m(x):
    '''
        Input: x (M x T x V x C) or (T x V x C)

        COCO: {0-nose 1-Leye 2-Reye 3-Lear 4Rear 5-Lsho 6-Rsho 7-Lelb 8-Relb 9-Lwri 10-Rwri 11-Lhip 12-Rhip 13-Lkne 14-Rkne 15-Lank 16-Rank}

        H36M:
        0: 'root',
        1: 'rhip',
        2: 'rkne',
        3: 'rank',
        4: 'lhip',
        5: 'lkne',
        6: 'lank',
        7: 'belly',
        8: 'neck',
        9: 'nose',
        10: 'head',
        11: 'lsho',
        12: 'lelb',
        13: 'lwri',
        14: 'rsho',
        15: 'relb',
        16: 'rwri'
    '''
    y = np.zeros(x.shape)
    y[..., 0, :] = (x[..., 11, :] + x[..., 12, :]) * 0.5
    y[..., 1, :] = x[..., 12, :]
    y[..., 2, :] = x[..., 14, :]
    y[..., 3, :] = x[..., 16, :]
    y[..., 4, :] = x[..., 11, :]
    y[..., 5, :] = x[..., 13, :]
    y[..., 6, :] = x[..., 15, :]
    y[..., 8, :] = (x[..., 5, :] + x[..., 6, :]) * 0.5
    y[..., 7, :] = (y[..., 0, :] + y[..., 8, :]) * 0.5
    y[..., 9, :] = x[..., 0, :]
    y[..., 10, :] = (x[..., 1, :] + x[..., 2, :]) * 0.5
    y[..., 11, :] = x[..., 5, :]
    y[..., 12, :] = x[..., 7, :]
    y[..., 13, :] = x[..., 9, :]
    y[..., 14, :] = x[..., 6, :]
    y[..., 15, :] = x[..., 8, :]
    y[..., 16, :] = x[..., 10, :]
    return y


def read_2dkeypoints_from_json(json_path, vid_size, scale_range, focus):
    with open(json_path, "r") as read_file:
        results = json.load(read_file)
    kpts_all = []
    for item in results:
        if focus != None and item['idx'] != focus:
            continue
        kpts = np.array(item['keypoints']).reshape([-1, 3])
        kpts_all.append(kpts)
    kpts_all = np.array(kpts_all)
    kpts_all = halpe2h36m(kpts_all)
    if vid_size:
        # 这条路经过
        w, h = vid_size
        scale = min(w, h) / 2.0
        kpts_all[:, :, :2] = kpts_all[:, :, :2] - np.array([w, h]) / 2.0
        kpts_all[:, :, :2] = kpts_all[:, :, :2] / scale
        motion = kpts_all
    if scale_range:
        # 这条路不经过
        motion = crop_scale(kpts_all, scale_range)
    return motion.astype(np.float32)


def read_2dkeypoints_from_numpy(numpy_data, vid_size, scale_range, focus, joints_format):
    '''
    :param numpy_data: [T,V,C=2或3]
    :param vid_size:
    :param scale_range:
    :param focus:
    :param joints_format: 关键点检测的格式，当前仅支持COCO和HALPE两种格式 最终转换成 H36M 的格式
    :return:
    '''
    if joints_format.lower() == 'coco':
        kpts_all = coco2h36m(numpy_data)
    elif joints_format.lower() == 'halpe':
        kpts_all = halpe2h36m(numpy_data)
    else:
        raise NotImplementedError

    if vid_size:
        # 这条路经过
        w, h = vid_size
        scale = min(w, h) / 2.0
        kpts_all[:, :, :2] = kpts_all[:, :, :2] - np.array([w, h]) / 2.0
        kpts_all[:, :, :2] = kpts_all[:, :, :2] / scale
        motion = kpts_all
    if scale_range:
        # 这条路不经过
        motion = crop_scale(kpts_all, scale_range)
    return motion.astype(np.float32)


class WildDetJsonDataset(Dataset):
    def __init__(self, json_path, clip_len=243, vid_size=None, scale_range=None, focus=None):
        self.json_path = json_path
        self.clip_len = clip_len
        self.vid_all = read_2dkeypoints_from_json(json_path, vid_size, scale_range, focus)

    def __len__(self):
        'Denotes the total number of samples'
        return math.ceil(len(self.vid_all) / self.clip_len)

    def __getitem__(self, index):
        'Generates one sample of data'
        st = index * self.clip_len
        end = min((index + 1) * self.clip_len, len(self.vid_all))
        return self.vid_all[st:end]


class WildDetNumpyDataset(Dataset):
    def __init__(self, numpy_data, clip_len=243, vid_size=None, scale_range=None, focus=None, format='coco'):
        self.numpy_data = numpy_data
        self.clip_len = clip_len
        # [T,V,C]
        self.vid_all = read_2dkeypoints_from_numpy(numpy_data, vid_size, scale_range, focus, format)

    def __len__(self):
        'Denotes the total number of samples'
        return math.ceil(len(self.vid_all) / self.clip_len)

    def total_frame(self):
        ':return T'
        return self.vid_all.shape[0]

    def __getitem__(self, index):
        'Generates one sample of data'
        st = index * self.clip_len
        end = min((index + 1) * self.clip_len, len(self.vid_all))
        cur_output = self.vid_all[st:end]
        cur_T = cur_output.shape[0]
        if cur_T < self.clip_len:
            pad_right = self.clip_len - cur_T
            cur_output = np.pad(cur_output, ((0, pad_right), (0, 0), (0, 0)), mode='edge')
        return cur_output


def read_input_ntu_json(json_path, vid_size, scale_range, focus):
    with open(json_path, "r") as read_file:
        results = json.load(read_file)
    kpts_all = []
    for item in results:
        # remove non-person picture
        if focus != None and item['idx'] != focus:
            continue
        kpts = np.array(item['keypoints']).reshape([-1, 3])
        kpts_all.append(kpts)
    kpts_all = np.array(kpts_all)
    # acquire h36m-style keypoints tensor [T,V,C]
    kpts_all = halpe2h36m(kpts_all)
    # [T,V,C]->[M=1,T,V,C]
    kpts_all = np.expand_dims(kpts_all, axis=0)
    return kpts_all.astype(dtype=np.float32)


def read_input_ntu_numpy(numpy_data, vid_size, scale_range, focus, joints_format):
    # [T,V,C] -> [M=1,T,V,C]
    numpy_data = numpy_data[np.newaxis, ...]
    if joints_format.lower() == 'coco':
        kpts_all = coco2h36m(numpy_data)
    elif joints_format.lower() == 'halpe':
        kpts_all = halpe2h36m(numpy_data)
    else:
        assert NotImplementedError, 'Currently,other keypoints formats are not supported!!'
    return kpts_all.astype(dtype=np.float32)


# referred by author https://github.com/Walter0807/MotionBERT/issues/23
class WildDetDatasetNTURGBD(Dataset):
    assert NotImplementedError, 'Please extend this class for single person or more than one person!!'


class WildDetDatasetNTURGBDSingle(WildDetDatasetNTURGBD):
    def __init__(self, numpy_data=None, json_path=None, clip_len=243, vid_size=None, scale_range=None, focus=None,
                 format='coco'):
        # convert keypoints in  h36m format，and transfer shape into [M=1,T,V,C]
        if numpy_data is not None:
            self.keypoint_3d = read_input_ntu_numpy(numpy_data, vid_size, scale_range, focus, format)
        elif json_path is not None:
            self.keypoint_3d = read_input_ntu_json(json_path, vid_size, scale_range, focus, format)
        else:
            assert NotImplementedError, 'Data source from numpy or json should not be None!!'

        self.json_path = json_path
        self.clip_len = clip_len
        self.vid_size = vid_size
        self.vid_h, self.vid_w = vid_size

        M, T, V, C = self.keypoint_3d.shape

        assert M == 1, 'Currently, this dataset only support one person!!'

        # resample for frame
        resample_id = np.linspace(0, T, num=self.clip_len, endpoint=False, dtype=int)
        # [M,T,V,C=2]
        keypoint_2d = self.keypoint_3d[..., :2]
        # [M,T,V,C=1]
        keypoint_conf = self.keypoint_3d[..., 2]
        keypoint_conf = keypoint_conf[..., None]

        # acquire input 2D tensor for correct range
        motion_cam = self.make_cam(x=keypoint_2d, img_shape=self.vid_size)
        motion = np.concatenate((motion_cam[:, resample_id], keypoint_conf[:, resample_id]), axis=-1)
        # for second person here we use zero padding here
        fake = np.zeros(motion.shape)
        # combine into a tensor
        motion = np.concatenate((motion, fake), axis=0)
        self.motions = [motion]

    def __getitem__(self, idx):
        return self.motions[idx].astype(dtype=np.float32)

    def __len__(self):
        return len(self.motions)

    # make input x from [0,1] to target range:[-1,1]
    def make_cam(self, x, img_shape):
        '''
            Input: x (M x T x V x C)
                   img_shape (height, width)
        '''
        h, w = img_shape
        if w >= h:
            x_cam = x / w * 2 - 1
        else:
            x_cam = x / h * 2 - 1
        return x_cam


class WildDetNumpyDatasetMeshRefine(Dataset):
    def __init__(self, motion_2d, rot_vec, body_shape, clip_len=243, vid_size=None, scale_range=None, focus=None,
                 format='coco'):
        self.numpy_data = motion_2d
        self.rot_vec = rot_vec
        self.body_shape = body_shape
        self.clip_len = clip_len
        # [T,V,C]
        self.vid_all = read_2dkeypoints_from_numpy(motion_2d, vid_size, scale_range, focus, format)

    def __len__(self):
        'Denotes the total number of samples'
        return math.ceil(len(self.vid_all) / self.clip_len)

    def total_frame(self):
        ':return T'
        return self.vid_all.shape[0]

    def __getitem__(self, index):
        'Generates one sample of data'
        st = index * self.clip_len
        end = min((index + 1) * self.clip_len, len(self.vid_all))
        cur_motion_2d_output = self.vid_all[st:end]
        cur_rot_vec_output = self.rot_vec[st:end]
        cur_body_shape_output = self.body_shape[st:end]
        cur_T = cur_motion_2d_output.shape[0]
        if cur_T < self.clip_len:
            pad_right = self.clip_len - cur_T
            cur_motion_2d_output = np.pad(cur_motion_2d_output, ((0, pad_right), (0, 0), (0, 0)), mode='edge')
            cur_rot_vec_output = np.pad(cur_rot_vec_output, ((0, pad_right), (0, 0), (0, 0)), mode='edge')
            cur_body_shape_output = np.pad(cur_body_shape_output, ((0, pad_right), (0, 0)), mode='edge')
        return {'motion2d': cur_motion_2d_output, 'rot_vec': cur_rot_vec_output, 'body_shape': cur_body_shape_output}
