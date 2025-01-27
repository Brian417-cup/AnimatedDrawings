import sys
import os.path as osp

sys.path.append(osp.dirname(__file__))

from math3d_SmartBody import dcm_from_axis, dcm2quat, quat_divide, quat2euler, quat2SMPLEuler
from bvh_helper_SmartBody import BvhNode, BvhHeader, write_bvh

import numpy as np
import numpy.typing as npt
from typing import Dict, Union, Tuple, List


class SMPL_SmartBodySkeleton(object):

    def __init__(self, smpl_out=True):
        # Joints names of SMPL / SMPLX follows https://github.com/softcat477/SMPL-to-FBX
        # What we modified with joint name is to remove underline
        self.root = 'Pelves'
        self.index2keypoint = {
            0: "Pelves",
            1: "Lhip",
            2: "Rhip",
            3: "Spine1",
            4: "LKnee",
            5: "RKnee",
            6: "Spine2",
            7: "LAnkle",
            8: "RAnkle",
            9: "Spine3",
            10: "LFoot",
            11: "RFoot",
            12: "Neck",
            13: "LCollar",
            14: "RCollar",
            15: "Head",
            16: "LShoulder",
            17: "RShoulder",
            18: "LElbow",
            19: "RElbow",
            20: "LWrist",
            21: "RWrist",
            22: "LHand",
            23: "RHand"
        }

        self.keypoint2index = {v: k for k, v in self.index2keypoint.items()}
        self.keypoint_num = len(self.index2keypoint)

        self.children = {
            self.index2keypoint[0]: [self.index2keypoint[1], self.index2keypoint[2], self.index2keypoint[3]],
            self.index2keypoint[1]: [self.index2keypoint[4]],
            self.index2keypoint[2]: [self.index2keypoint[5]],
            self.index2keypoint[3]: [self.index2keypoint[6]],
            self.index2keypoint[4]: [self.index2keypoint[7]],
            self.index2keypoint[5]: [self.index2keypoint[8]],
            self.index2keypoint[6]: [self.index2keypoint[9]],
            self.index2keypoint[7]: [self.index2keypoint[10]],
            self.index2keypoint[8]: [self.index2keypoint[11]],
            self.index2keypoint[9]: [self.index2keypoint[12], self.index2keypoint[14], self.index2keypoint[13]],
            self.index2keypoint[10]: [],
            self.index2keypoint[11]: [],
            self.index2keypoint[12]: [self.index2keypoint[15]],
            self.index2keypoint[13]: [self.index2keypoint[16]],
            self.index2keypoint[14]: [self.index2keypoint[17]],
            self.index2keypoint[15]: [],
            self.index2keypoint[16]: [self.index2keypoint[18]],
            self.index2keypoint[17]: [self.index2keypoint[19]],
            self.index2keypoint[18]: [self.index2keypoint[20]],
            self.index2keypoint[19]: [self.index2keypoint[21]],
            self.index2keypoint[20]: [self.index2keypoint[22]],
            self.index2keypoint[21]: [self.index2keypoint[23]],
            self.index2keypoint[22]: [],
            self.index2keypoint[23]: []
        }

        self.parent = {self.root: None}
        for parent, children in self.children.items():
            for child in children:
                self.parent[child] = parent

        self.left_joints = [
            joint for joint in self.keypoint2index
            if joint.startswith('L')
        ]
        self.right_joints = [
            joint for joint in self.keypoint2index
            if joint.startswith('R')
        ]

        # SmartBody坐标系(Y向上，Z向前，X向右)下的T-pose
        self.initial_directions = {
            self.index2keypoint[0]: [0, 0, 0],
            self.index2keypoint[2]: [-1, 0, 0],
            self.index2keypoint[5]: [0, -1, 0],
            self.index2keypoint[8]: [0, -1, 0],
            self.index2keypoint[11]: [0, 0, 1],
            self.index2keypoint[1]: [1, 0, 0],
            self.index2keypoint[4]: [0, -1, 0],
            self.index2keypoint[7]: [0, -1, 0],
            self.index2keypoint[10]: [0, 0, 1],
            self.index2keypoint[3]: [0, 1, 0],
            self.index2keypoint[6]: [0, 1, 0],
            self.index2keypoint[9]: [0, 1, 0],
            self.index2keypoint[14]: [-1, 1, 0],
            self.index2keypoint[13]: [1, 1, 0],
            self.index2keypoint[12]: [0, 1, 0],
            self.index2keypoint[15]: [0, 1, 0],
            self.index2keypoint[16]: [1, 0, 0],
            self.index2keypoint[18]: [1, 0, 0],
            self.index2keypoint[20]: [1, 0, 0],
            self.index2keypoint[22]: [1, 0, 0],
            self.index2keypoint[17]: [-1, 0, 0],
            self.index2keypoint[19]: [-1, 0, 0],
            self.index2keypoint[21]: [-1, 0, 0],
            self.index2keypoint[23]: [-1, 0, 0]
        }

        self.smpl_out = smpl_out

    def get_initial_offset(self, poses_3d):
        # TODO: RANSAC
        bone_lens = {self.root: [0]}
        stack = [self.root]
        while stack:
            parent = stack.pop()
            p_idx = self.keypoint2index[parent]
            for child in self.children[parent]:
                # define endsite of some chains
                if child in (
                        self.index2keypoint[11], self.index2keypoint[10],
                        self.index2keypoint[23], self.index2keypoint[22],
                        self.index2keypoint[15]):
                    bone_lens[child] = 0.4 * bone_lens[parent]
                    continue
                stack.append(child)

                c_idx = self.keypoint2index[child]
                bone_lens[child] = np.linalg.norm(
                    poses_3d[:, p_idx] - poses_3d[:, c_idx],
                    axis=1
                )

        bone_len = {}
        for joint in self.keypoint2index:
            if joint.startswith('L') or joint.startswith('R'):
                base_name = joint.replace('L', '').replace('R', '')
                left_len = np.mean(bone_lens['L' + base_name])
                right_len = np.mean(bone_lens['R' + base_name])
                bone_len[joint] = (left_len + right_len) / 2
            else:
                bone_len[joint] = np.mean(bone_lens[joint])

        initial_offset: Dict[str, npt.NDArray[np.float32]] = {}
        for joint, direction in self.initial_directions.items():
            direction = np.array(direction) / max(np.linalg.norm(direction), 1e-12)
            initial_offset[joint] = direction * bone_len[joint]

        initial_offset[self.index2keypoint[0]] = (
                initial_offset[self.index2keypoint[11]] + initial_offset[self.index2keypoint[8]] +
                initial_offset[self.index2keypoint[5]] + initial_offset[self.index2keypoint[2]] +
                initial_offset[self.index2keypoint[0]])
        initial_offset[self.index2keypoint[0]] *= -1

        return initial_offset

    def get_bvh_header(self, poses_3d):
        '''
        Attention, the rotation order is zxy generated for bvh format
        Args:
            poses_3d: [T,V,C]

        Returns:

        '''
        initial_offset: Dict[str, npt.NDArray[np.float32]] = self.get_initial_offset(poses_3d)

        # The BvhNode includes relative attributes of *.bvh file
        nodes: Dict[str, BvhNode] = {}

        # This is to traverse dict values
        for joint in self.index2keypoint.values():
            is_root = joint == self.root
            # is_end_site = 'EndSite' in joint
            is_end_site = joint in (
                self.index2keypoint[11], self.index2keypoint[10],
                self.index2keypoint[23], self.index2keypoint[22],
                self.index2keypoint[15])

            if self.smpl_out:
                nodes[joint] = BvhNode(
                    name=joint,
                    offset=initial_offset[joint],
                    # this format of rotation implies the SMPL
                    rotation_order='xyz',
                    is_root=is_root,
                    is_end_site=is_end_site,
                )
            else:
                nodes[joint] = BvhNode(
                    name=joint,
                    offset=initial_offset[joint],
                    rotation_order='zxy' if not is_end_site else '',
                    is_root=is_root,
                    is_end_site=is_end_site,
                )

        for joint, children in self.children.items():
            nodes[joint].children = [nodes[child] for child in children]
            for child in children:
                nodes[child].parent = nodes[joint]

        header = BvhHeader(root=nodes[self.root], nodes=nodes)
        return header

    def pose2euler(self, pose, header):
        '''

        Args:
            pose: [V,C]
            header:

        Returns:

        '''
        channel = []
        quats = {}
        eulers = {}
        stack = [header.root]
        while stack:
            node = stack.pop()
            joint = node.name
            joint_idx = self.keypoint2index[joint]

            if node.is_root:
                channel.extend(pose[joint_idx])

            index = self.keypoint2index
            order = None

            ###################################################
            # following transfer containing first annotated and the other transfered

            if joint == self.index2keypoint[0]:
                x_dir = pose[1] - pose[2]
                y_dir = pose[3] - pose[joint_idx]
                z_dir = None
                order = 'yzx'

            # elif joint == 'Hips':
            #     x_dir = pose[index['LeftUpLeg']] - pose[index['RightUpLeg']]
            #     y_dir = pose[index['Spine']] - pose[joint_idx]
            #     z_dir = None
            #     order = 'yzx'

            elif joint in [self.index2keypoint[2], self.index2keypoint[5]]:
                child_idx = self.keypoint2index[node.children[0].name]
                x_dir = pose[0] - pose[2]
                y_dir = pose[joint_idx] - pose[child_idx]
                z_dir = None
                order = 'yzx'

            # elif joint in ['RightUpLeg', 'RightLeg']:
            #     child_idx = self.index2keypoint[node.children[0].name]
            #     x_dir = pose[index['Hips']] - pose[index['RightUpLeg']]
            #     y_dir = pose[joint_idx] - pose[child_idx]
            #     z_dir = None
            #     order = 'yzx'

            elif joint in [self.index2keypoint[1], self.index2keypoint[4]]:
                child_idx = self.keypoint2index[node.children[0].name]
                x_dir = pose[1] - pose[0]
                y_dir = pose[joint_idx] - pose[child_idx]
                z_dir = None
                order = 'yzx'

            # elif joint in ['LeftUpLeg', 'LeftLeg']:
            #     child_idx = self.index2keypoint[node.children[0].name]
            #     x_dir = pose[index['LeftUpLeg']] - pose[index['Hips']]
            #     y_dir = pose[joint_idx] - pose[child_idx]
            #     z_dir = None
            #     order = 'yzx'

            elif joint in [self.index2keypoint[3], self.index2keypoint[6], self.index2keypoint[9]]:
                child_idx = self.keypoint2index[node.children[0].name]
                x_dir = pose[1] - pose[2]
                y_dir = pose[child_idx] - pose[joint_idx]
                z_dir = None
                order = 'yzx'

            # elif joint == 'Spine':
            #     x_dir = pose[index['LeftUpLeg']] - pose[index['RightUpLeg']]
            #     y_dir = pose[index['Spine3']] - pose[joint_idx]
            #     z_dir = None
            #     order = 'yzx'

            elif joint == self.index2keypoint[12]:
                x_dir = pose[16] - pose[17]
                y_dir = pose[joint_idx] - pose[9]
                z_dir = None
                order = 'yzx'

            # elif joint == 'Spine3':
            #     x_dir = pose[index['LeftArm']] - \
            #             pose[index['RightArm']]
            #     y_dir = pose[joint_idx] - pose[index['Spine']]
            #     z_dir = None
            #     order = 'yzx'

            # ##########################################################
            # This is not used in SMPL
            # elif joint == 'Neck':
            #     x_dir = None
            #     z_dir = pose[index['Spine3']] - pose[joint_idx]
            #     y_dir = pose[index['Head']] - pose[index['Spine3']]
            #     order = 'yxz'
            # ##########################################################

            elif joint == self.index2keypoint[16]:
                x_dir = pose[18] - pose[joint_idx]
                y_dir = None
                z_dir = pose[18] - pose[20]
                order = 'xyz'

            # elif joint == 'LeftArm':
            #     x_dir = pose[index['LeftForeArm']] - pose[joint_idx]
            #     y_dir = None
            #     z_dir = pose[index['LeftForeArm']] - pose[index['LeftHand']]
            #     order = 'xyz'

            elif joint == self.index2keypoint[18]:
                x_dir = pose[20] - pose[joint_idx]
                y_dir = None
                z_dir = pose[joint_idx] - pose[16]
                order = 'xyz'

            # elif joint == 'LeftForeArm':
            #     x_dir = pose[index['LeftHand']] - pose[joint_idx]
            #     y_dir = None
            #     z_dir = pose[joint_idx] - pose[index['LeftArm']]
            #     order = 'xyz'

            elif joint == self.index2keypoint[17]:
                x_dir = pose[joint_idx] - pose[19]
                y_dir = None
                z_dir = pose[19] - pose[21]
                order = 'xyz'

            # elif joint == 'RightArm':
            #     x_dir = pose[joint_idx] - pose[index['RightForeArm']]
            #     y_dir = None
            #     z_dir = pose[index['RightForeArm']] - pose[index['RightHand']]
            #     order = 'xyz'

            elif joint == self.index2keypoint[19]:
                x_dir = pose[joint_idx] - pose[21]
                y_dir = None
                z_dir = pose[joint_idx] - pose[17]
                order = 'xyz'

            # elif joint == 'RightForeArm':
            #     x_dir = pose[joint_idx] - pose[index['RightHand']]
            #     y_dir = None
            #     z_dir = pose[joint_idx] - pose[index['RightArm']]
            #     order = 'xyz'

            ################################################################
            # Extra joints in SMPL for 14 and 13 indicies of joints
            elif joint in (self.index2keypoint[14], self.index2keypoint[13]):
                x_dir = pose[16] - pose[17]
                y_dir = pose[joint_idx] - pose[9]
                z_dir = None
                order = 'xzy'

            if order:
                dcm = dcm_from_axis(x_dir, y_dir, z_dir, order)
                quats[joint] = dcm2quat(dcm)
            else:
                quats[joint] = quats[self.parent[joint]].copy()

            local_quat = quats[joint].copy()
            if node.parent:
                local_quat = quat_divide(
                    q=quats[joint], r=quats[node.parent.name]
                )

            euler = quat2euler(
                q=local_quat, order=node.rotation_order
            )
            euler = np.rad2deg(euler)
            eulers[joint] = euler
            channel.extend(euler)

            for child in node.children[::-1]:
                if not child.is_end_site:
                    stack.append(child)

        return channel

    def pose2SMPLEuler(self, pose, header):
        '''
        This is merely to dealt once a time called
        Args:
            pose: [V,C] ndarray
            header:

        Returns:
            Like SMPL degrees in radians
        '''
        channel: List[List[npt.NDArray[np.float32]]] = []
        quats: Dict[str, npt.NDArray[np.float32]] = {}
        eulers_smpl: Dict[int, npt.NDArray[np.float32]] = {}

        stack = [header.root]
        while stack:
            node = stack.pop()
            joint = node.name
            joint_idx = self.keypoint2index[joint]

            if node.is_root:
                channel.extend(pose[joint_idx])

            # index = self.keypoint2index
            order = None

            ###################################################
            # following transfer containing first annotated and the other transfered

            if joint == self.index2keypoint[0]:
                x_dir = pose[1] - pose[2]
                y_dir = pose[3] - pose[joint_idx]
                z_dir = None
                order = 'yzx'

            # elif joint == 'Hips':
            #     x_dir = pose[index['LeftUpLeg']] - pose[index['RightUpLeg']]
            #     y_dir = pose[index['Spine']] - pose[joint_idx]
            #     z_dir = None
            #     order = 'yzx'

            elif joint in [self.index2keypoint[2], self.index2keypoint[5]]:
                child_idx = self.keypoint2index[node.children[0].name]
                x_dir = pose[0] - pose[2]
                y_dir = pose[joint_idx] - pose[child_idx]
                z_dir = None
                order = 'yzx'

            # elif joint in ['RightUpLeg', 'RightLeg']:
            #     child_idx = self.index2keypoint[node.children[0].name]
            #     x_dir = pose[index['Hips']] - pose[index['RightUpLeg']]
            #     y_dir = pose[joint_idx] - pose[child_idx]
            #     z_dir = None
            #     order = 'yzx'

            elif joint in [self.index2keypoint[1], self.index2keypoint[4]]:
                child_idx = self.keypoint2index[node.children[0].name]
                x_dir = pose[1] - pose[0]
                y_dir = pose[joint_idx] - pose[child_idx]
                z_dir = None
                order = 'yzx'

            # elif joint in ['LeftUpLeg', 'LeftLeg']:
            #     child_idx = self.index2keypoint[node.children[0].name]
            #     x_dir = pose[index['LeftUpLeg']] - pose[index['Hips']]
            #     y_dir = pose[joint_idx] - pose[child_idx]
            #     z_dir = None
            #     order = 'yzx'

            elif joint in [self.index2keypoint[3], self.index2keypoint[6], self.index2keypoint[9]]:
                child_idx = self.keypoint2index[node.children[0].name]
                x_dir = pose[1] - pose[2]
                y_dir = pose[child_idx] - pose[joint_idx]
                z_dir = None
                order = 'yzx'

            # elif joint == 'Spine':
            #     x_dir = pose[index['LeftUpLeg']] - pose[index['RightUpLeg']]
            #     y_dir = pose[index['Spine3']] - pose[joint_idx]
            #     z_dir = None
            #     order = 'yzx'

            elif joint == self.index2keypoint[12]:
                x_dir = pose[16] - pose[17]
                y_dir = pose[joint_idx] - pose[9]
                z_dir = None
                order = 'yzx'

            # elif joint == 'Spine3':
            #     x_dir = pose[index['LeftArm']] - \
            #             pose[index['RightArm']]
            #     y_dir = pose[joint_idx] - pose[index['Spine']]
            #     z_dir = None
            #     order = 'yzx'

            # ##########################################################
            # This is not used in SMPL
            # elif joint == 'Neck':
            #     x_dir = None
            #     z_dir = pose[index['Spine3']] - pose[joint_idx]
            #     y_dir = pose[index['Head']] - pose[index['Spine3']]
            #     order = 'yxz'
            # ##########################################################

            elif joint == self.index2keypoint[16]:
                x_dir = pose[18] - pose[joint_idx]
                y_dir = None
                z_dir = pose[18] - pose[20]
                order = 'xyz'

            # elif joint == 'LeftArm':
            #     x_dir = pose[index['LeftForeArm']] - pose[joint_idx]
            #     y_dir = None
            #     z_dir = pose[index['LeftForeArm']] - pose[index['LeftHand']]
            #     order = 'xyz'

            elif joint == self.index2keypoint[18]:
                x_dir = pose[20] - pose[joint_idx]
                y_dir = None
                z_dir = pose[joint_idx] - pose[16]
                order = 'xyz'

            # elif joint == 'LeftForeArm':
            #     x_dir = pose[index['LeftHand']] - pose[joint_idx]
            #     y_dir = None
            #     z_dir = pose[joint_idx] - pose[index['LeftArm']]
            #     order = 'xyz'

            elif joint == self.index2keypoint[17]:
                x_dir = pose[joint_idx] - pose[19]
                y_dir = None
                z_dir = pose[19] - pose[21]
                order = 'xyz'

            # elif joint == 'RightArm':
            #     x_dir = pose[joint_idx] - pose[index['RightForeArm']]
            #     y_dir = None
            #     z_dir = pose[index['RightForeArm']] - pose[index['RightHand']]
            #     order = 'xyz'

            elif joint == self.index2keypoint[19]:
                x_dir = pose[joint_idx] - pose[21]
                y_dir = None
                z_dir = pose[joint_idx] - pose[17]
                order = 'xyz'

            # elif joint == 'RightForeArm':
            #     x_dir = pose[joint_idx] - pose[index['RightHand']]
            #     y_dir = None
            #     z_dir = pose[joint_idx] - pose[index['RightArm']]
            #     order = 'xyz'

            ################################################################
            # Extra joints in SMPL for 14 and 13 indicies of joints
            elif joint in (self.index2keypoint[14], self.index2keypoint[13]):
                x_dir = pose[16] - pose[17]
                y_dir = pose[joint_idx] - pose[9]
                z_dir = None
                order = 'xzy'

            if order:
                dcm = dcm_from_axis(x_dir, y_dir, z_dir, order)
                quats[joint] = dcm2quat(dcm)
            else:
                quats[joint] = quats[self.parent[joint]].copy()

            local_quat = quats[joint].copy()
            if node.parent:
                local_quat = quat_divide(
                    q=quats[joint], r=quats[node.parent.name]
                )

            euler_radians = quat2SMPLEuler(
                q=local_quat, order=node.rotation_order
            )

            eulers_smpl[self.keypoint2index[joint]] = euler_radians

            channel.extend(euler_radians)

            # This demonstragte that end site joints are not considered in standard *.bvh file's generation
            for child in node.children[::-1]:
                if not child.is_end_site:
                    stack.append(child)

        # temporal end site dealing way
        eulers_smpl[23] = np.zeros_like(eulers_smpl[21])
        eulers_smpl[22] = np.zeros_like(eulers_smpl[20])
        eulers_smpl[15] = np.zeros_like(eulers_smpl[12])
        eulers_smpl[11] = np.zeros_like(eulers_smpl[8])
        eulers_smpl[10] = np.zeros_like(eulers_smpl[7])

        # sort dict by joint index in order to align with output's correct order
        eulers_smpl = {k: eulers_smpl[k] for k in sorted(eulers_smpl)}

        return channel, eulers_smpl

    def poses2bvh(self, poses_3d, header=None, output_file=None):
        if not header:
            header = self.get_bvh_header(poses_3d)

        channels = []
        for frame, pose in enumerate(poses_3d):
            channels.append(self.pose2euler(pose, header))

        if output_file:
            write_bvh(output_file, header, channels)

        return channels, header

    def pose2smpl(self, pose_3d)->npt.NDArray[np.float32]:
        '''

        Args:
            pose_3d: [T,V,C=3]

        Returns:
            eular_3d: [T,V,C=3]

        '''
        assert pose_3d.shape[-1] == 3, 'Input must be 3 channels!!'

        final_eular_smpl = np.zeros_like(pose_3d)

        header_smpl = self.get_bvh_header(pose_3d)
        channels = []

        for frame, pose in enumerate(pose_3d):
            channel_smpl, eular_smpl = self.pose2SMPLEuler(pose, header_smpl)
            channels.append(channel_smpl)
            eular_smpl = np.stack([item for item in eular_smpl.values()], axis=0)
            final_eular_smpl[frame] = eular_smpl

        print('final smpl eulars', final_eular_smpl.shape, 'input 3d pose', pose_3d.shape)

        return final_eular_smpl


# convert h36m 17 joint to smpl 24 joint
def convert_h36m17_to_smpl(joints_h36m: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    '''

    Args:
        joints_h36m: [N,V_h36m,C] or [N,T,V_h36m,C]

    Returns:
        joints_smpl: [N,V_smpl,C] or [N,T,V_smpl,C]
    '''
    # [N,T,V,C]
    if len(joints_h36m.shape) == 4:
        N, T, V, C = joints_h36m.shape
        joints_smpl = np.zeros([N, T, 24, C], dtype=np.float32)
    elif len(joints_h36m.shape) == 3:
        N, V, C = joints_h36m.shape
        joints_smpl = np.zeros([N, 24, C], dtype=np.float32)
    else:
        raise NotImplementedError

    joints_smpl[..., 2, :] = joints_h36m[..., 1, :]
    joints_smpl[..., 5, :] = joints_h36m[..., 2, :]
    joints_smpl[..., 8, :] = joints_h36m[..., 3, :]
    joints_smpl[..., 1, :] = joints_h36m[..., 4, :]
    joints_smpl[..., 4, :] = joints_h36m[..., 5, :]
    joints_smpl[..., 7, :] = joints_h36m[..., 6, :]
    joints_smpl[..., 6, :] = joints_h36m[..., 7, :]
    joints_smpl[..., 0, :] = (joints_h36m[..., 7, :] + joints_h36m[..., 6, :]) / 2
    joints_smpl[..., 12, :] = joints_h36m[..., 8, :]
    joints_smpl[..., 15, :] = (joints_h36m[..., 9, :] + joints_h36m[..., 8, :]) / 2
    joints_smpl[..., 17, :] = joints_h36m[..., 14, :]
    joints_smpl[..., 19, :] = joints_h36m[..., 15, :]
    joints_smpl[..., 21, :] = joints_h36m[..., 16, :]
    joints_smpl[..., 16, :] = joints_h36m[..., 11, :]
    joints_smpl[..., 18, :] = joints_h36m[..., 12, :]
    joints_smpl[..., 20, :] = joints_h36m[..., 13, :]

    joints_smpl[..., 11, :] = joints_h36m[..., 3, :] + 1 / 4.0 * (joints_h36m[..., 3, :] - joints_h36m[..., 2, :])
    joints_smpl[..., 10, :] = joints_h36m[..., 6, :] + 1 / 4.0 * (joints_h36m[..., 6, :] - joints_h36m[..., 5, :])
    joints_smpl[..., 9, :] = 1 / 6.0 * joints_h36m[..., 8, :] + 5 / 6.0 * joints_h36m[..., 7, :]
    joints_smpl[..., 23, :] = joints_h36m[..., 16, :] + 1 / 4.0 * (joints_h36m[..., 16, :] - joints_h36m[..., 15, :])
    joints_smpl[..., 22, :] = joints_h36m[..., 13, :] + 1 / 4.0 * (joints_h36m[..., 13, :] - joints_h36m[..., 12, :])

    joints_smpl[..., 3, :] = (joints_smpl[..., 0, :] + joints_smpl[..., 6, :]) / 2
    joints_smpl[..., 14, :] = (joints_smpl[..., 9, :] + joints_smpl[..., 17, :]) / 2
    joints_smpl[..., 13, :] = (joints_smpl[..., 9, :] + joints_smpl[..., 16, :]) / 2

    return joints_smpl


if __name__ == '__main__':
    t = SMPL_SmartBodySkeleton()
    a = {1: 5}
