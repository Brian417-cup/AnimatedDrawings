import os
import os.path as osp
import sys

sys.path.append(osp.join(osp.dirname(__file__)))

import numpy as np
import numpy.typing as npt
import cv2
from deparser.joint import Joint
from deparser.bvh import BVH
from deparser.vectors import Vectors
from deparser.transform import Transform
from typing import List, Dict, Tuple, Union
from copy import deepcopy
from scipy.spatial.transform import Rotation


# 针对bvh做简单二维投影可视化
class BVHVisualizer():
    def __init__(self, bvh_path: str):
        self.bvh_path = bvh_path
        # 这里拿到的bvh默认初始状态不是T字形状态，而是第一帧状态
        # init_root_rot_eular = np.array([0.0, 90.0, 0.0] and rotation order for root joint = 'ZXY'
        # self.bvh = BVH.from_file(osp.join(bvh_path), init_root_rot_eular=np.array([0.0, 90.0, 0.0]))
        self.bvh = BVH.from_file(osp.join(bvh_path))
        self.bvh_name = self.bvh.name

        # 和节点相关的参数
        # 按bvh的顺序去存储所有的结点名(其中末节点一定是有多个的)
        self.all_joint_names: List[str] = self.bvh.get_joint_names()
        self.joint_num: int = self.bvh.joint_num
        self.root_joint = self.bvh.root_joint

        # 构建从根节点开始的所有节点名
        self.chain_joint_names_from_root: List[List[str]] = self.bvh.chain_joint_names_fron_root
        # 构建从根节点开始的所有节点名和其对应的下标，不同的链路分别存储在不同的列表内
        self.chain_joint_names_to_idx_dict: List[Dict[str, int]] = self.bvh.chain_joint_names_to_idx_dict_from_root

        # 获取和时间相关的参数
        self.total_frame = self.bvh.frame_max_num
        self.fps: float = self.bvh.frame_time * 1000

    def visualize_animated_orthogonal_projection(self,
                                                 canvas_w: int, canvas_h: int,
                                                 offset_x: int, offset_y: int,
                                                 scale: float):
        '''
        简单的正交投影来实现二维可视化
        :param canvas_w: 画布宽
        :param canvas_h: 画布高
        :param offset_x: x方向偏移量
        :param offset_y: y方向偏移量
        :param scale: 缩放系数
        :return:
        '''
        self.canvas_w = canvas_w
        self.canvas_h = canvas_h

        # 所有帧的关键点数据以第一帧的根节点为原点,并设置相应的缩放系数
        self.bvh.apply_frame(frame_num=0)
        start_root_pos: Vectors = self.root_joint.get_world_position()
        self.bvh.offset(-start_root_pos)
        self.bvh.set_scale(scale=scale)

        # 获取动画间隔
        delay = self._get_delay_from_fps(fps=self.fps)

        for t in range(self.total_frame):
            # 准备空白画布
            blank_canvas = self._create_blank_canvas(w=canvas_w, h=canvas_h)

            # 启用当前帧
            self.bvh.apply_frame(frame_num=t)

            # [V*C=3],给当前的帧数据再加上一定的偏移量
            all_frame_data: npt.NDArray[np.float32] = self.root_joint.get_chain_worldspace_positions()
            all_frame_data += np.tile(np.array([offset_x, offset_y, 0.0], dtype=np.float32), reps=(self.joint_num))

            # 画点
            # 这里采用简单的正交投影(直接去掉z轴)
            joint_x, joint_y = all_frame_data[0::3], all_frame_data[1::3]
            for x, y in zip(joint_x, joint_y):
                # 因为画板二维平面的y轴方向和实际坐标系的方向相反，这里要先做取反y轴的方向
                cv2.circle(blank_canvas, (int(x), int(canvas_h - y)), radius=3, color=(0, 0, 255), thickness=-1)

            # 画线
            for chain_idx, cur_sequence in enumerate(self.chain_joint_names_from_root):
                for idx in range(1, len(cur_sequence)):
                    # 获取链路上首尾的结点名
                    start_name = cur_sequence[idx - 1]
                    end_name = cur_sequence[idx]
                    # 获取链路上首尾的结点名所对应的下标
                    start_idx = self.chain_joint_names_to_idx_dict[chain_idx][start_name]
                    end_idx = self.chain_joint_names_to_idx_dict[chain_idx][end_name]

                    # 根据索引拿到正交投影后二维的点数据
                    cv2.line(blank_canvas, pt1=(int(joint_x[start_idx]), int(canvas_h - joint_y[start_idx])),
                             pt2=(int(joint_x[end_idx]), int(canvas_h - joint_y[end_idx])),
                             color=(255, 0, 0), thickness=2)

            # 绘制
            cv2.imshow('res', blank_canvas)
            cv2.waitKey(delay)

    # 创建空白画布
    def _create_blank_canvas(self, w: int, h: int):
        return np.ones((h, w, 3), dtype=np.uint8) * 255

    # 根据fps计算相邻帧的时间间隔
    def _get_delay_from_fps(self, fps: float):
        return int(1000 / fps)

    def visualize_init_pose_orthogonal_projection(self, canvas_w: int, canvas_h: int,
                                                  offset_x: int, offset_y: int,
                                                  scale: float):
        '''
        简单的正交投影来实现初始姿态的二维可视化
        :param canvas_w: 画布宽
        :param canvas_h: 画布高
        :param offset_x: x方向偏移量
        :param offset_y: y方向偏移量
        :param scale: 缩放系数
        :return:
        '''
        self.bvh.apply_frame(frame_num=-1)  # self.bvh.apply_init_local_offset()
        self.bvh.set_scale(scale=scale)
        start_root_pos: Vectors = self.root_joint.get_world_position()
        self.bvh.offset(-start_root_pos)

        # 获得初始化帧的数据
        init_frame_data: npt.NDArray[np.float32] = np.array(self.root_joint.get_chain_worldspace_positions())
        init_frame_data += np.tile(np.array([offset_x, offset_y, 0.0], dtype=np.float32), reps=(self.joint_num))

        # 准备空白画布
        blank_canvas = self._create_blank_canvas(w=canvas_w, h=canvas_h)

        # 画点
        # 这里采用简单的正交投影(直接去掉z轴)
        joint_x, joint_y = init_frame_data[0::3], init_frame_data[1::3]
        for x, y in zip(joint_x, joint_y):
            # 因为画板二维平面的y轴方向和实际坐标系的方向相反，这里要先做取反y轴的方向
            cv2.circle(blank_canvas, (int(x), int(canvas_h - y)), radius=3, color=(0, 0, 255), thickness=-1)

        # 画线
        for chain_idx, cur_sequence in enumerate(self.chain_joint_names_from_root):
            for idx in range(1, len(cur_sequence)):
                # 获取链路上首尾的结点名
                start_name = cur_sequence[idx - 1]
                end_name = cur_sequence[idx]
                # 获取链路上首尾的结点名所对应的下标
                start_idx = self.chain_joint_names_to_idx_dict[chain_idx][start_name]
                end_idx = self.chain_joint_names_to_idx_dict[chain_idx][end_name]

                # 根据索引拿到正交投影后二维的点数据
                cv2.line(blank_canvas, pt1=(int(joint_x[start_idx]), int(canvas_h - joint_y[start_idx])),
                         pt2=(int(joint_x[end_idx]), int(canvas_h - joint_y[end_idx])),
                         color=(255, 0, 0), thickness=2)

        # 绘制
        cv2.imshow('res', blank_canvas)
        cv2.waitKey(-1)


if __name__ == '__main__':
    bvh_path = osp.join(osp.join(r"../../", 'output/TEST_14/bvh/TEST_14_ik.bvh'))
    visualizer = BVHVisualizer(bvh_path=bvh_path)

    # 做所有帧的动画
    visualizer.visualize_animated_orthogonal_projection(canvas_w=600, canvas_h=600, offset_x=120, offset_y=100,
                                                        scale=4.0)

    # 做初始帧的姿态查看
    visualizer.visualize_init_pose_orthogonal_projection(canvas_w=600, canvas_h=600, offset_x=200, offset_y=200,
                                                         scale=4.0)
