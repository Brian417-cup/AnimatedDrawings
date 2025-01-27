import os
import torch
import torch.nn.functional as F
import numpy as np
import copy


# 对坐标做缩放，并将范围转换到[-1,1]中
def crop_scale(motion, scale_range=[1, 1]):
    '''
        Motion: [(M), T, 17, 3]. [M,T,V,C]
        Normalize to [-1, 1]
    '''
    result = copy.deepcopy(motion)
    # 在这里conf_z=0则表示这个人是通过zero_padding填充而来的
    # valid_coords的shape为 [M'(表示有效人数),T,V,C'=2]
    valid_coords = motion[motion[..., 2] != 0][:, :2]
    if len(valid_coords) < 4:
        return np.zeros(motion.shape)
    xmin = min(valid_coords[:, 0])
    xmax = max(valid_coords[:, 0])
    ymin = min(valid_coords[:, 1])
    ymax = max(valid_coords[:, 1])
    # 选择一个随即缩放比
    ratio = np.random.uniform(low=scale_range[0], high=scale_range[1], size=1)[0]
    # 获得最终的缩放比例
    scale = max(xmax - xmin, ymax - ymin) * ratio
    # 如果缩放系数为全0时，则选择最终返回全0的数
    if scale == 0:
        return np.zeros(motion.shape)
    xs = (xmin + xmax - scale) / 2
    ys = (ymin + ymax - scale) / 2
    result[..., :2] = (motion[..., :2] - [xs, ys]) / scale
    # 将坐标转换到[-1,1]
    result[..., :2] = (result[..., :2] - 0.5) * 2
    result = np.clip(result, -1, 1)
    return result


def crop_scale_3d(motion, scale_range=[1, 1]):
    '''
        Motion: [T, 17, 3]. (x, y, z)
        Normalize to [-1, 1]
        Z is relative to the first frame's root.
    '''
    result = copy.deepcopy(motion)
    result[:, :, 2] = result[:, :, 2] - result[0, 0, 2]
    xmin = np.min(motion[..., 0])
    xmax = np.max(motion[..., 0])
    ymin = np.min(motion[..., 1])
    ymax = np.max(motion[..., 1])
    ratio = np.random.uniform(low=scale_range[0], high=scale_range[1], size=1)[0]
    scale = max(xmax - xmin, ymax - ymin) / ratio
    if scale == 0:
        return np.zeros(motion.shape)
    xs = (xmin + xmax - scale) / 2
    ys = (ymin + ymax - scale) / 2
    result[..., :2] = (motion[..., :2] - [xs, ys]) / scale
    result[..., 2] = result[..., 2] / scale
    result = (result - 0.5) * 2
    return result


def flip_data(data):
    """
    horizontal flip
        data: [N, F, 17, D] or [F, 17, D]. X (horizontal coordinate) is the first channel in D.
    Return
        result: same
    """
    left_joints = [4, 5, 6, 11, 12, 13]
    right_joints = [1, 2, 3, 14, 15, 16]
    flipped_data = copy.deepcopy(data)
    flipped_data[..., 0] *= -1  # flip x of all joints
    flipped_data[..., left_joints + right_joints, :] = flipped_data[..., right_joints + left_joints, :]
    return flipped_data


def flip_thetas_batch(thetas):
    """Flip thetas in batch.

    Parameters
    ----------
    thetas : numpy.array
        Joints in shape (N, F, num_thetas*3)
    theta_pairs : list
        List of theta pairs.

    Returns
    -------
    numpy.array
        Flipped thetas with shape (N, F, num_thetas*3)

    """
    # Joint pairs which defines the pairs of joint to be swapped when the image is flipped horizontally.
    theta_pairs = ((1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21), (22, 23))
    thetas_flip = copy.deepcopy(thetas).reshape(*thetas.shape[:2], 24, 3)
    # reflect horizontally
    thetas_flip[:, :, :, 1] = -1 * thetas_flip[:, :, :, 1]
    thetas_flip[:, :, :, 2] = -1 * thetas_flip[:, :, :, 2]
    # change left-right parts
    for pair in theta_pairs:
        thetas_flip[:, :, pair[0], :], thetas_flip[:, :, pair[1], :] = \
            thetas_flip[:, :, pair[1], :], thetas_flip[:, :, pair[0], :].clone()

    return thetas_flip.reshape(*thetas.shape[:2], -1)


# 重采样函数，针对ActionDataset数据集时使用
# 返回的shape为 [target_len]
# 这里需要根据是否随机采样做分类讨论，再根据原始帧长和目标帧长的大小关系继续做分类讨论
def resample(ori_len, target_len, replay=False, randomness=True):
    '''
    @param randomness: 在训练时为True，在推理时为False
    '''
    if replay:
        # 如果目标采取帧小于原始总帧长，从多余的部分随机选择一个起始点，再取target_len长度
        if ori_len > target_len:
            st = np.random.randint(ori_len - target_len)
            return range(st, st + target_len)  # Random clipping from sequence
        # 如果目标帧大于原始总帧长，从原始帧长中不断地循环播放，直到padding满目标帧长
        else:
            return np.array(range(target_len)) % ori_len  # Replay padding
    else:
        # 在训练阶段的dataset中有效
        if randomness:
            # 设置采样点
            even = np.linspace(0, ori_len, num=target_len, endpoint=False)
            # 如果原始帧长小于目标帧长，采取帧重复来做补帧策略
            # 表示对采样点做 floor和ceil操作，最终通过sel做加权求和来获得最终的res采样点
            if ori_len < target_len:
                low = np.floor(even)
                high = np.ceil(even)
                # 随机生成[0,2)的even.shape个整数
                sel = np.random.randint(2, size=even.shape)
                result = np.sort(sel * low + (1 - sel) * high)
            # 如果原始帧长大于目标帧长
            else:
                # 获取采样间隔
                interval = even[1] - even[0]
                # 再在原来的线性采样基础上做随机扰动
                result = np.random.random(even.shape) * interval + even
            # 不管最后原始帧比目标帧长还是短，最终需要通过np.clip来做最终范围的收缩确定
            result = np.clip(result, a_min=0, a_max=ori_len - 1).astype(np.uint32)
        # 如果不随机，则采用线性采样
        else:
            result = np.linspace(0, ori_len, num=target_len, endpoint=False, dtype=int)
        return result


def split_clips(vid_list, n_frames, data_stride):
    result = []
    n_clips = 0
    st = 0
    i = 0
    saved = set()
    while i < len(vid_list):
        i += 1
        if i - st == n_frames:
            result.append(range(st, i))
            saved.add(vid_list[i - 1])
            st = st + data_stride
            n_clips += 1
        if i == len(vid_list):
            break
        if vid_list[i] != vid_list[i - 1]:
            if not (vid_list[i - 1] in saved):
                resampled = resample(i - st, n_frames) + st
                result.append(resampled)
                saved.add(vid_list[i - 1])
            st = i
    return result


if __name__ == '__main__':
    # 学习一下resample这个函数里的原理
    target_len = 30
    origin_len = 50
    # 关于 np.linspace 这个函数
    # 参数为 start,end,num=sample_cnt,endpoint=True/False (表示是  [start,end] 还是 [start,end))
    even = np.linspace(0, origin_len, num=target_len, endpoint=False)
    print(even)
    if origin_len < target_len:
        low = np.floor(even)
        high = np.ceil(even)
        print(low, high)
        # np.random.randint(end,size=total_shape) 这个函数表示是从 [0,end) 中选择出 total_shape尺寸的随机整数
        sel = np.random.randint(2, size=even.shape)
        print(sel)
        res = np.sort(sel * low + (1 - sel) * high)
        print(res)
    else:
        # 获取采样频率
        interval = even[1] - even[0]
        res = np.random.random(even.shape) * interval + even
        print(res)
    res = np.clip(res, a_min=0, a_max=origin_len - 1).astype(np.uint32)
    print(res)
