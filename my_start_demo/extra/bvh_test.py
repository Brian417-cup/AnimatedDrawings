import os
import os.path as osp
import numpy as np
import numpy.typing as npt
import cv2
from animated_drawings.model.bvh import BVH
from scipy.spatial.transform import Rotation
from typing import List, Dict, Tuple

# 模拟使用bvh文件做读取

bvh = BVH.from_file(bvh_fn=osp.join("../2_smartbody.bvh"))
all_joint_nodes: List[str] = bvh.get_joint_names()
print('所有的节点名字', bvh.get_joint_names())

# 设置缩放系数
bvh.set_scale(scale=1.0)
root_pos = bvh.root_joint.get_world_position()
print(root_pos.shape)
# 以根节点构造原点
bvh.offset(-bvh.root_joint.get_world_position())
# 遍历每一帧
for i in range(bvh.frame_max_num):
    bvh.apply_frame(frame_num=i)

    l = bvh.root_joint.get_chain_worldspace_positions()
    print(l)

# 关于旋转其实这里可以用这个库来实现
a = np.array([[1.0, 2.0, 0.0], [2.0, 2.0, 0.0]])
r_matrix = Rotation.from_euler(seq='xyz', angles=[0.0, 0.0, 90.0], degrees=True)
a = r_matrix.apply(a)
print(a)
