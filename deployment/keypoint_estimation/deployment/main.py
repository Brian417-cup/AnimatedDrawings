from mmpose.apis.inference import _inference_single_pose_model, \
    inference_top_down_pose_model, process_mmdet_results, init_pose_model
from mmdet.apis import inference_detector, init_detector
from mmpose.apis import vis_pose_result
from mmcv import Config
import os
import glob
import numpy as np
import mmcv

cfg = Config.fromfile(os.path.join('sketch_config', 'config.py'))
cfg.pose_checkpoint = os.path.join('sketch_weight', 'best_AP_epoch_72.pth')
pose_model = init_pose_model(cfg, cfg.pose_checkpoint)


def start_prediction_wihout_det(file_path: str, img_height: int, img_width):
    detect_box = [{'bbox': np.array([0, 0, img_height, img_width])}]
    pose_results, returned_output = inference_top_down_pose_model(pose_model, file_path, detect_box, format='xyxy')
    print(pose_results)
    vis_pose_result(pose_model, file_path, pose_results, show=True)


if __name__ == '__main__':
    for item in glob.glob(os.path.join('data', '*.png')):
        img = mmcv.imread(item)
        start_prediction_wihout_det(file_path=item, img_height=img.shape[1], img_width=img.shape[0])
