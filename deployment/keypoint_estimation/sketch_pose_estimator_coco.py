import onnxruntime
import cv2
import os.path as osp
import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import copy
from top_down_eval import keypoints_from_heatmaps
import numpy as np
from collections import OrderedDict

# To align，here we adopt to 15 keypoints for following:
DrawBoardCOCOIndex2Name = {
    0: 'nose',
    # 脖子这个点是由left_shoudler和right_shoulder共同计算得到的
    # 1: 'neck',
    5: 'left_shoulder',
    6: 'right_shoulder',
    7: 'left_elbow',
    8: 'right_elbow',
    9: 'left_wrist',
    10: 'right_wrist',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle',
    # 这个hip_mid和center_point是手动算出来的,hip_mid这个关键点在openpose中本身是有的，
    # 但是在openmmlab提供的算法库中屏蔽了这个点，所以要通过中点做插值计算
    # -1: 'hip_mid',
}


class DrawBoardPoseEstimatorCOCO():
    def __init__(self,
                 model_path: str,
                 consider_index: list = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]):
        self.model_path = model_path
        self.session = onnxruntime.InferenceSession(model_path)
        self.output_tesnor_names = [node.name for node in self.session.get_outputs()]
        self.input_tensor_names = self.session.get_inputs()
        self.fine_width = 192
        self.fine_height = 256
        self.consider_index = consider_index

    # 根据图片数据做预测
    def estimate_by_img_data(self, img_data: np.array, use_filter=True, need_show=False):
        self.img_data = img_data
        # 前处理
        init_h, init_w, _ = img_data.shape
        img_tensor = cv2.dnn.blobFromImage(img_data, scalefactor=1.0 / 255,
                                           size=(self.fine_width, self.fine_height),
                                           mean=[0.485, 0.456, 0.406],
                                           swapRB=True,
                                           crop=False)

        # 推理
        heatmap = self.session.run(self.output_tesnor_names,
                                   input_feed={self.input_tensor_names[0].name: img_tensor})[0]

        # 后处理
        detection_frame_x, detection_frame_y = 0, 0
        center_point_xy = np.array([[detection_frame_x + init_w * 0.5, detection_frame_y + init_h * 0.5]],
                                   dtype=np.float32)
        scale = np.array([[init_w / 200, init_h / 200]], dtype=np.float32)
        res = keypoints_from_heatmaps(heatmap, center_point_xy, scale)[0]
        # 获取第一个简笔画小人所有待考虑的关键点坐标
        self.all_single_res = copy.deepcopy(np.array(res.tolist()[0]))
        self.all_single_res = self.all_single_res.astype(np.int64)
        # 将shape转换为 [N,2],N为所有待考虑的关键点
        self.filtered_single_res = np.array(res.tolist()[0])[self.consider_index]

        for i, point in enumerate(self.filtered_single_res):
            x, y = point
            self.filtered_single_res[i] = (x, y)

        self.filtered_single_res = self.filtered_single_res.astype(np.int64)

        if need_show:
            self.vis_pose_with_joint_dict(img_data, need_show=need_show)

        return self.filtered_single_res

    # 根据图片路径做预测
    def estimate_by_img_path(self, img_path: str, use_filter=True, need_show: bool = False):
        img_data = cv2.imread(img_path)
        self.filtered_single_res = self.estimate_by_img_data(img_data, use_filter, need_show)
        return self.filtered_single_res

    # 根据  dict{关键点名:(x,y)}  可视化关键点
    def vis_pose_with_joint_dict(self, need_show: bool = False):
        # for i, point in enumerate(self.filtered_single_res):
        #     x, y = point
        #     cv2.circle(self.img_data, (x, y), 4, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

        print(f'total node cnt is {len(self.tresult)}')

        for node_name, point in self.tresult.items():
            print(f'current node name is {node_name}')
            x, y = point
            cv2.circle(self.img_data, (x, y), 4, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

        if need_show:
            cv2.imshow('vis_result', self.img_data)
            cv2.waitKey(-1)
        return self.img_data

    # 根据关键点构建 {关键点名:(x,y)} 这样的字典
    def build_points_to_dict(self):
        tresult = OrderedDict()
        for index in self.consider_index:
            node_name = DrawBoardCOCOIndex2Name[index]
            tresult[node_name] = self.all_single_res[index]

        # 添加两个额外的点 neck 和 hip_mid
        left_shoulder_x, left_shoulder_y = tresult['left_shoulder']
        right_shoulder_x, right_shoulder_y = tresult['right_shoulder']
        tresult['neck'] = [(left_shoulder_x + right_shoulder_x) // 2, (left_shoulder_y + right_shoulder_y) // 2]

        left_hip_x, left_hip_y = tresult['left_hip']
        right_hip_x, right_hip_y = tresult['right_hip']
        tresult['hip_mid'] = [(left_hip_x + right_hip_x) // 2, (left_hip_y + right_hip_y) // 2]

        self.tresult = tresult

        return tresult

    # 为骨骼添加中心点
    def add_centerpoint(self):
        thorax = self.tresult['neck']
        pelvis = self.tresult['hip_mid']
        x = (thorax[0] + pelvis[0]) // 2
        y = (thorax[1] + pelvis[1]) // 2
        self.tresult['centerpoint'] = [x, y]

        print('根节点添加成功！！')

        return self.tresult


if __name__ == '__main__':
    model_path = osp.join('../../weight/doodle_pose/human_skeleton.onnx')
    drawboard_pose_estimator = DrawBoardPoseEstimatorCOCO(model_path=model_path)
    drawboard_pose_estimator.estimate_by_img_path(img_path='test2.png', use_filter=False)
    drawboard_pose_estimator.build_points_to_dict()
    drawboard_pose_estimator.vis_pose_with_joint_dict(need_show=True)
