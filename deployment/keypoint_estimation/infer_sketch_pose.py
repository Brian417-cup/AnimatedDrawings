'''
python infer_sketch_pose.py --img_path <complete path for sketch image path>  --model_path <complete onnx path for keypoint estimator>
'''
import copy
import os
import os.path as osp
import sys

sys.path.append(osp.join(osp.dirname(osp.abspath(__file__))))

from collections import OrderedDict
from typing import List
import cv2
import numpy as np
import onnxruntime as ort
from top_down_eval import keypoints_from_heatmaps


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=r'deployment/output/sketch_estimator.onnx',
                        help='the onnx model path')
    parser.add_argument('--img_path', type=str, default=r'deployment/data/test1.png', help='the test img path')
    args = parser.parse_args()
    return args


SKETCH_INDEX_2_NAME = {
    0: 'nose',
    # neck = (left_shoulder+right_shoulder)/2
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
    # hip_mid= (left_hip+right_hip)/2
    # -1: 'hip_mid',
}


class SketchPoseEstimator():
    def __init__(self,
                 model_path: str,
                 consider_keypoint_indicies: List[int] = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]):
        assert osp.exists(model_path), 'Model path should exist!!'
        self.model_path = model_path
        self.session = ort.InferenceSession(model_path)
        self.output_tesnor_names = [node.name for node in self.session.get_outputs()]
        self.input_tensor_names = self.session.get_inputs()
        self.fine_width = 192
        self.fine_height = 256

        self.consider_keypoint_indicies = consider_keypoint_indicies

    def preprocess(self, img_data):
        '''

        Args:
            img_data: [H,W,C]

        Returns:
            img_tensor: [N=1,C,H,W] with Normalize and ToTensor()
        '''
        init_h, init_w, _ = img_data.shape
        img_tensor = cv2.dnn.blobFromImage(img_data, scalefactor=1.0 / 255,
                                           size=(self.fine_width, self.fine_height),
                                           mean=[0.485, 0.456, 0.406],
                                           swapRB=True,
                                           crop=False)
        return img_tensor

    def inference(self, img_path, bounding_bbox: List[int] = None, show_res=False):
        '''

        Args:
            img_path: str, path for image
            bounding_bbox: List[int], in [x1,y1,x2,y2] format. Default is None.
                           If exists, it will use for only given range for keypoints estimator.
            show_res: bool, if True, the result will be drawn with initial image data.

        Returns:
            keypoints_res: [V,C=2]
            joint_name_to_xy_dict: Dict { joint_name: (x,y) }
        '''
        assert osp.exists(img_path), 'Image path should exist!!'
        self.img_path = img_path
        img_data = cv2.imread(img_path)

        # check if bounding bbox exists
        if bounding_bbox is not None:
            l, t, r, b = [round(x) for x in bounding_bbox]
            img_data = img_data[t:b, l:r]

        self.img_data = img_data
        img_tensor = self.preprocess(img_data)
        heatmap = self.session.run(self.output_tesnor_names,
                                   input_feed={self.input_tensor_names[0].name: img_tensor})[0]

        keypoints_res, joint_name_to_xy_dict = self.postprocess(img_data, heatmap, show_res=show_res)

        return keypoints_res, joint_name_to_xy_dict

    def postprocess(self, img_data, heatmap, show_res=False):
        '''

        Args:
            img_data: [H,W,C]
            heatmap: [N,V,out_h,out_W]

        Returns:
            filtered_single_res: [V,C=2]
            joint_name_to_xy_dict: { joint_name: (x,y) }
        '''
        init_h, init_w, _ = img_data.shape
        detection_frame_x, detection_frame_y = 0, 0
        center_point_xy = np.array([[detection_frame_x + init_w * 0.5, detection_frame_y + init_h * 0.5]],
                                   dtype=np.float32)
        scale = np.array([[init_w / 200, init_h / 200]], dtype=np.float32)
        res = keypoints_from_heatmaps(heatmap, center_point_xy, scale)[0]
        # acquire all keypoints from first sketch
        self.all_single_res = copy.deepcopy(np.array(res.tolist()[0]))
        self.all_single_res = self.all_single_res.astype(np.int64)
        # [V,2]
        self.filtered_single_res = np.array(res.tolist()[0])[self.consider_keypoint_indicies]

        for i, point in enumerate(self.filtered_single_res):
            x, y = point
            self.filtered_single_res[i] = (x, y)

        # transfer float32 -> int32 for each keypoint's coordinate
        self.filtered_single_res = self.filtered_single_res.astype(np.int64)

        # store joint_name to xy in a dict {joint_name: (x,y)}
        self.joint_name_to_xy_dict = self._build_points_to_dict()

        if show_res:
            self.vis_pose_with_joint_dict()

        return self.filtered_single_res, self.joint_name_to_xy_dict

    def _build_points_to_dict(self):
        '''

        Returns: tresult dict { joint_name: (x,y) }

        '''
        tresult = OrderedDict()
        for index in self.consider_keypoint_indicies:
            node_name = SKETCH_INDEX_2_NAME[index]
            tresult[node_name] = self.all_single_res[index]

        # add neck and hip_mid by middle insert
        left_shoulder_x, left_shoulder_y = tresult['left_shoulder']
        right_shoulder_x, right_shoulder_y = tresult['right_shoulder']
        tresult['neck'] = [(left_shoulder_x + right_shoulder_x) // 2, (left_shoulder_y + right_shoulder_y) // 2]

        left_hip_x, left_hip_y = tresult['left_hip']
        right_hip_x, right_hip_y = tresult['right_hip']
        tresult['hip_mid'] = [(left_hip_x + right_hip_x) // 2, (left_hip_y + right_hip_y) // 2]

        return tresult

    def vis_pose_with_joint_dict(self):
        assert self.img_data is not None, 'Image data should not be None!!'
        assert self.joint_name_to_xy_dict is not None, 'Joint name to concrrect coordinate (x,y) should not be None!!'
        print(f'>> Total node cnt is {len(self.joint_name_to_xy_dict)}')

        drawn_img_data = self.img_data

        for node_name, point in self.joint_name_to_xy_dict.items():
            print(f'>> Current node name is {node_name}')
            x, y = point
            cv2.circle(drawn_img_data, (x, y), 4, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

        cv2.imshow('sketch_vis_result', drawn_img_data)
        cv2.waitKey(-1)
        return drawn_img_data


if __name__ == '__main__':
    args = get_args()
    model_path = args.model_path
    img_path = args.img_path
    sketch_estimator = SketchPoseEstimator(model_path=model_path)
    sketch_estimator.inference(img_path=img_path, show_res=True)
