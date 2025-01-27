import argparse
import os
import os.path as osp
import sys

sys.path.append(osp.join(osp.dirname(__file__), '..'))
from deployment.detection import SketchDetector
from deployment.keypoint_estimation import SketchPoseEstimator
import cv2
from typing import Dict, List
import numpy as np
import numpy.typing as npt
import yaml


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--detector_path', type=str, default='detection/deployment/output/sketch_detector.onnx')
    parser.add_argument('--estimator_path', type=str,
                        default='keypoint_estimation/deployment/output/sketch_estimator.onnx')
    parser.add_argument('--img_path', type=str, default='keypoint_estimation/deployment/data/test1.png')
    parser.add_argument('--mor_ite', type=int, default=5,
                        help='Iteration for dialation and close operation when acquire mask.')
    args = parser.parse_args()
    return args


class SketchDetectorAndEstimator():
    def __init__(self, estimator_path: str, detector_path: str = None):
        '''

        Args:
            estimator_path: *.onnx file for estimator. It should be not None.
            detector_path: *.onnx file for detector. If it is None, the final process for inference will be not used.
        '''
        if detector_path is not None:
            assert osp.exists(detector_path), 'Path for detector should exist!!'
            self.detector = SketchDetector(model_path=detector_path)
        else:
            self.detector = None

        assert osp.exists(estimator_path), 'Path for estimator should exist!!'
        self.estimator = SketchPoseEstimator(model_path=estimator_path)

    def inference(self, img_path: str, show_res: bool = False,
                  need_mask: bool = False, morphops_iteration: int = 15,
                  out_dir: str = 'output'):
        '''

        Args:
            img_path: Source path for image.
            show_res: If show results ,final prediction will be visualized in a temporal window. Else, not shown.
            need_mask: If need mask, the output will use relative morphops operation to get mask.
            morphops_iteration: If need_mask=True, result will use morphops.
            out_dir: Base dir for saving dealt image data and config files.

        Returns:

        '''
        if self.detector is not None:
            detector_res: List[Dict[str, npt.NDArray[np.float32]]] = self.detector.inference(img_path=img_path,
                                                                                             show_res=show_res)
            # [x1,y1,x2,y2]
            bbox = detector_res[0]['bbox']

        # [V,C=2], Dict { joint_name: (x,y) }
        keypoints_res, joint_name_to_xy_dict = self.estimator.inference(img_path=img_path, bounding_bbox=bbox,
                                                                        show_res=show_res)
        # write result
        self._write_result(img_path, out_dir, bbox=bbox, joint_name_to_xy_dict=joint_name_to_xy_dict,
                           need_mask=need_mask, show_mask=show_res, morphops_iteration=morphops_iteration)

    def _write_result(self, img_path: str, out_dir: str, bbox: List[int] = None,
                      joint_name_to_xy_dict: Dict[str, npt.NDArray[np.int32]] = Dict,
                      need_mask: bool = True, morphops_iteration: int = 15, show_mask: bool = False):
        img_data = cv2.imread(img_path)
        if bbox is not None:
            img_h, img_w = bbox[-1] - bbox[1], bbox[-2] - bbox[0]
        else:
            img_h, img_w = img_data.shape[:2]

        img_h, img_w = int(img_h), int(img_w)

        skeleton = []
        # certainfy root joint
        skeleton.append({'loc': [round(x) for x in joint_name_to_xy_dict['hip_mid']], 'name': 'root', 'parent': None})
        skeleton.append({'loc': [round(x) for x in joint_name_to_xy_dict['hip_mid']], 'name': 'hip', 'parent': 'root'})
        # other joints
        skeleton.append({'loc': [round(x) for x in joint_name_to_xy_dict['neck']], 'name': 'torso', 'parent': 'hip'})
        skeleton.append({'loc': [round(x) for x in joint_name_to_xy_dict['nose']], 'name': 'nose', 'parent': 'torso'})
        skeleton.append({'loc': [round(x) for x in joint_name_to_xy_dict['right_shoulder']], 'name': 'right_shoulder',
                         'parent': 'torso'})
        skeleton.append({'loc': [round(x) for x in joint_name_to_xy_dict['right_elbow']], 'name': 'right_elbow',
                         'parent': 'right_shoulder'})
        skeleton.append({'loc': [round(x) for x in joint_name_to_xy_dict['right_wrist']], 'name': 'right_hand',
                         'parent': 'right_elbow'})
        skeleton.append({'loc': [round(x) for x in joint_name_to_xy_dict['left_shoulder']], 'name': 'left_shoulder',
                         'parent': 'torso'})
        skeleton.append({'loc': [round(x) for x in joint_name_to_xy_dict['left_elbow']], 'name': 'left_elbow',
                         'parent': 'left_shoulder'})
        skeleton.append({'loc': [round(x) for x in joint_name_to_xy_dict['left_wrist']], 'name': 'left_hand',
                         'parent': 'left_elbow'})
        skeleton.append(
            {'loc': [round(x) for x in joint_name_to_xy_dict['right_hip']], 'name': 'right_hip', 'parent': 'root'})
        skeleton.append({'loc': [round(x) for x in joint_name_to_xy_dict['right_knee']], 'name': 'right_knee',
                         'parent': 'right_hip'})
        skeleton.append({'loc': [round(x) for x in joint_name_to_xy_dict['right_ankle']], 'name': 'right_foot',
                         'parent': 'right_knee'})
        skeleton.append(
            {'loc': [round(x) for x in joint_name_to_xy_dict['left_hip']], 'name': 'left_hip', 'parent': 'root'})
        skeleton.append(
            {'loc': [round(x) for x in joint_name_to_xy_dict['left_knee']], 'name': 'left_knee', 'parent': 'left_hip'})
        skeleton.append({'loc': [round(x) for x in joint_name_to_xy_dict['left_ankle']], 'name': 'left_foot',
                         'parent': 'left_knee'})

        img_name, img_suffix = osp.basename(img_path).split('.')[0], osp.basename(img_path).split('.')[-1]
        output_base_dir = osp.join(out_dir, img_name)
        os.makedirs(output_base_dir, exist_ok=True)
        char_cfg_path = osp.join(output_base_dir, 'character_config.yaml')
        cropped_img_path = osp.join(output_base_dir, 'cropped_image.' + img_suffix)
        with open(char_cfg_path, 'w') as f:
            char_cfg = {'skeleton': skeleton, 'height': img_h, 'width': img_w}
            yaml.dump(char_cfg, f)

            if bbox is not None:
                l, t, r, b = [round(x) for x in bbox]
                img_data = img_data[t:b, l:r]

            cv2.imwrite(cropped_img_path, img_data)

        self.char_cfg_path, self.cropped_img_path = char_cfg_path, cropped_img_path
        print('>> Successfully save config file for character in', char_cfg_path, '!!')
        print('>> Successfully save cropped image file for character in', cropped_img_path, '!!')

        if need_mask:
            img_mask = self._get_mask(img_data=img_data, show_res=show_mask, morphops_iteration=morphops_iteration)

            mask_img_path = osp.join(output_base_dir, 'mask_image.' + img_suffix)

            cv2.imwrite(mask_img_path, img_mask)
            print('>> Successfully save segmented image file for character in', mask_img_path, '!!')
            self.mask_img_path = mask_img_path

    def _get_mask(self, img_data: npt.NDArray[np.uint8], show_res: bool = False, morphops_iteration=12):
        from skimage import measure
        from scipy import ndimage

        # find adaptive threshold
        img_data = np.min(img_data, axis=2)
        img_data = cv2.adaptiveThreshold(img_data, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 8)
        img_data = cv2.bitwise_not(img_data)

        # morphops operation
        # here, iteration for close and dialte is hyperparameter
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        img_data = cv2.morphologyEx(img_data, cv2.MORPH_CLOSE, kernel, iterations=morphops_iteration)
        img_data = cv2.morphologyEx(img_data, cv2.MORPH_DILATE, kernel, iterations=morphops_iteration)

        # floodfill (generate mask in the center and with padding for left, right, top and bottom)
        mask = np.zeros([img_data.shape[0] + 2, img_data.shape[1] + 2], np.uint8)
        mask[1:-1, 1:-1] = img_data.copy()

        # img_floodfill is the results of floodfill. Starts off all white
        img_data_floodfill = np.full(img_data.shape, 255, np.uint8)

        # choose 10 points along each image side. Use them as seed for floodfill
        h, w = img_data.shape[:2]
        for x in range(0, w - 1, 10):
            cv2.floodFill(img_data_floodfill, mask, (x, 0), 0)
            cv2.floodFill(img_data_floodfill, mask, (x, h - 1), 0)

        for y in range(0, h - 1, 10):
            cv2.floodFill(img_data_floodfill, mask, (0, y), 0)
            cv2.floodFill(img_data_floodfill, mask, (w - 1, y), 0)

        # remove the edge that exists character. It will influence for contour finding
        img_data_floodfill[0, :] = img_data_floodfill[-1, :] = img_data_floodfill[:, 0] = img_data_floodfill[:, -1] = 0

        # acquire largest contour
        mask2 = cv2.bitwise_not(img_data_floodfill)
        final_mask = None
        biggest_size = 0

        contours = measure.find_contours(mask2, 0.0)
        for c in contours:
            x = np.zeros(mask2.T.shape, np.uint8)
            cv2.fillPoly(x, [np.int32(c)], 1)
            size = len(np.where(x == 1)[0])
            if size > biggest_size:
                final_mask = x
                biggest_size = size

        if final_mask is None:
            msg = 'Found no contours within image'
            assert False, msg

        final_mask = ndimage.binary_fill_holes(final_mask).astype(int)
        final_mask = 255 * final_mask.astype(np.uint8)

        final_mask = final_mask.T

        if show_res:
            cv2.imshow('mask', final_mask)
            cv2.waitKey(-1)

        return final_mask


if __name__ == '__main__':
    args = get_args()
    sketch_executor = SketchDetectorAndEstimator(
        detector_path=args.detector_path,
        estimator_path=args.estimator_path
    )
    sketch_executor.inference(img_path=args.img_path, show_res=True, need_mask=True, morphops_iteration=args.mor_ite)
