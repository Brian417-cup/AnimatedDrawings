import numpy as np
import numpy.typing as npt
import onnxruntime
import torch
from typing import Tuple
from scipy.spatial.transform import Rotation as R


class PoseEstimator():
    def __init__(self, from_main_cfg=None, model_path: str = ''):
        assert model_path.endswith('.onnx'), 'In onnx inference mode, you should offer correct *.onnx file!!'
        self.main_cfg = from_main_cfg
        self.sesstion = onnxruntime.InferenceSession(model_path)

    def __call__(self, x: Tuple[npt.NDArray, torch.Tensor]):
        '''

        :param x: [N,T,V,C] for 2D pose sequence
        :return: [N,T,V,C] for 3D pose sequence
        '''
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        output_tuple = self.sesstion.run(['pose3d'], {'pose2d': x.astype(np.float32)}, )
        out_pose_3d = output_tuple[0]
        return out_pose_3d


class ActionRecognizer():
    def __init__(self, from_main_cfg=None, model_path: str = ''):
        assert model_path.endswith('.onnx'), 'In onnx inference mode, you should offer correct *.onnx file!!'
        self.main_cfg = from_main_cfg
        self.sesstion = onnxruntime.InferenceSession(model_path)

    def __call__(self, x: Tuple[npt.NDArray, torch.Tensor]):
        '''

        :param x: [N,M=2,T,V,C] for 2D pose sequence
        :return: [N,action_label] for action label
        '''
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        output_tuple = self.sesstion.run(['action_label'], {'pose2d': x.astype(np.float32)}, )
        out_pose_3d = output_tuple[0]
        return out_pose_3d


class MeshRefine():
    def __init__(self, from_main_cfg=None, model_path: str = ''):
        assert model_path.endswith('.onnx'), 'In onnx inference mode, you should offer correct *.onnx file!!'
        self.main_cfg = from_main_cfg
        self.sesstion = onnxruntime.InferenceSession(model_path)

    def __call__(self, x: Tuple[npt.NDArray, torch.Tensor],
                 rot_vec: Tuple[npt.NDArray, torch.Tensor],
                 body_shape: Tuple[npt.NDArray, torch.Tensor]):
        '''
        The current inference is for standard SMPL model
        :param x: [N,T,V,C] for 2D pose sequence
        :param rot_vec: [N,T,V,C] for 2D pose sequence
        :param body_shape: [N,T,10] for 2D pose sequence
        :return:
                rot_vec: [N,T,V,3],
                shape: [N,T,10],
                vertices: [N,T,6890,3],
                keypoint_3d: [N,T,V,C=3]
        '''
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()

        N, T = x.shape[:2]

        if isinstance(rot_vec, torch.Tensor):
            rot_vec = rot_vec.detach().cpu().numpy()
        rot_vec = rot_vec.reshape([N, T, -1])

        if isinstance(body_shape, torch.Tensor):
            body_shape = body_shape.detach().cpu().numpy()

        output_tuple = self.sesstion.run(['rot_mat', 'shape', 'vertices', 'keypoint_3d'],
                                         {'pose2d': x.astype(np.float32),
                                          'hybrik_theta': rot_vec.astype(np.float32),
                                          'hybrik_shape': body_shape.astype(np.float32)}, )
        rot_mat, shape, vertices, keypoint_3d = output_tuple

        rot_vec = R.from_matrix(rot_mat.reshape([-1, 3, 3])).as_rotvec().reshape([N, T, -1, 3])

        return rot_vec, shape, vertices, keypoint_3d
