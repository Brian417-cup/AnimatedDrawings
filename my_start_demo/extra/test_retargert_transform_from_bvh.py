import copy
import math
import os.path as osp
import sys

import numpy as np
import numpy.typing as npt

sys.path.append(osp.join(osp.dirname(__file__), '../..'))

from animated_drawings.model.bvh import BVH
from animated_drawings.model.vectors import Vectors
from animated_drawings.model.quaternions import Quaternions
from typing import List, Union, Tuple, Dict
from sklearn.decomposition import PCA

# This function will help us directrly choose following frames
x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)


def test_bvh_retarget():
    motion_cfg_file = osp.join(r"../..", 'examples/config/motion/jumping_jacks.yaml')
    motion_file = osp.join('../..', 'examples/bvh/cmu1/jumping_jacks.bvh')
    bvh = BVH.from_file(motion_file)

    bvh_joint_names = bvh.get_joint_names()
    print('all joint names are ', bvh_joint_names)

    forward_perp_joint_group_names = [
        ('LeftShoulder', 'RightShoulder'),
        ('LeftHip', 'RightHip')
    ]

    skeleton_fwd: Vectors = bvh.get_skeleton_fwd(forward_perp_joint_group_names)
    q: Quaternions = Quaternions.rotate_between_vectors(skeleton_fwd, Vectors([1, 0, 0]))
    print('quaternion result is :', q)

    from scipy.spatial.transform import Rotation as R
    eular_result = R.from_quat(q.qs).as_euler(seq='xyz')
    print('corresponding eular result is :', eular_result)

    bvh.rotation_offset(q)

    scale_factor = 0.025
    bvh.set_scale(scale_factor)
    bvh.offset(bvh.root_joint.get_world_position())

    groundplane_joint_name = 'LeftAnkle'
    groundplane_joint = bvh.root_joint.get_transform_by_name(groundplane_joint_name)
    bvh_groundplane_y = groundplane_joint.get_world_position()[1]
    print('ground plane y is ', bvh_groundplane_y)

    bvh.offset(np.array([0, -bvh_groundplane_y, 0]))

    ################################################
    # For each frame we should also rotate forward vector that is +x
    # compute normalized joint positions and fwd vectors
    # position world space
    # [T,V*3]
    joint_positions: npt.NDArray[np.float32] = np.empty([bvh.frame_max_num, 3 * bvh.joint_num], dtype=np.float32)
    # [T,3]
    fwd_vectors: npt.NDArray[np.float32] = np.empty([bvh.frame_max_num, 3], dtype=np.float32)
    for frame_idx in range(bvh.frame_max_num):
        bvh.apply_frame(frame_idx)
        joint_positions[frame_idx] = bvh.root_joint.get_chain_worldspace_positions()
        fwd_vectors[frame_idx] = bvh.get_skeleton_fwd(forward_perp_joint_group_names).vs[0]

    # get relative position
    # [T,3]
    bvh_root_positions: npt.NDArray[np.float32] = joint_positions[:, :3]
    # [T,V*3]
    joint_positions = joint_positions - np.tile(bvh_root_positions, [1, len(bvh_joint_names)])

    v1 = np.tile(np.array([1.0, 0.0], dtype=np.float32), reps=(bvh.frame_max_num, 1))
    v2 = fwd_vectors
    dot: npt.NDArray[np.float32] = v1[:, 0] * v2[:, 0] + v1[:, 1] * v2[:, 2]
    det: npt.NDArray[np.float32] = v1[:, 0] * v2[:, 2] - v2[:, 0] * v1[:, 1]
    # [T,2]
    angle: npt.NDArray[np.float32] = np.arctan2(det, dot).astype(np.float32)
    # angle result is in radians
    angle %= 2 * np.pi
    # here, we use a mode-like degree suffix to get all larger than 0 angle
    angle = np.where(angle < 0.0, angle + 2 * np.pi, angle)

    # rotate joint in nonhomogeneous way around y axis
    # for each frame, rotation matrix is like:
    # (
    #   cos(angle[frame_idx])  0  sin(angle[frame_idx])
    #            0             1        0
    #   -sin(angle[frame_idx]) 0  cos(angle[frame_idx])
    # )

    # [T,V*3]
    joint_positions_copy = copy.deepcopy(joint_positions)
    all_rotation_matrix = []
    for frame_idx in range(bvh.frame_max_num):
        rot_mat = np.identity(3).astype(np.float32)
        rot_mat[0, 0] = math.cos(angle[frame_idx])
        rot_mat[0, 2] = math.sin(angle[frame_idx])
        rot_mat[2, 0] = -math.sin(angle[frame_idx])
        rot_mat[2, 2] = math.cos(angle[frame_idx])

        # [3,3] @ [3,V] -> [3,V]
        rotated_joints: npt.NDArray[np.float32] = rot_mat @ joint_positions[frame_idx].reshape([-1, 3]).T
        # [3, V] -> [V,3]
        joint_positions[frame_idx] = rotated_joints.T.reshape(joint_positions[frame_idx].shape)

        all_rotation_matrix.append(rot_mat)

    all_rotation_matrix = np.stack(all_rotation_matrix, axis=0)

    #####################################################
    # another quick way to rotation built with help of scipy library
    # from scipy.spatial.transform import Rotation as R
    # y_rotation_matrix = R.from_euler(seq='y', angles=angle).as_matrix()
    #
    # print('rotation matrix transfer 2 ways:', np.isclose(y_rotation_matrix, all_rotation_matrix).all())
    #
    # batch_result = np.einsum(
    #     'tcc,tvc->tcv',
    #     # 'tcc,tvc->tvc',
    #     y_rotation_matrix,
    #     joint_positions_copy.reshape([bvh.frame_max_num, -1, 3])
    # ).transpose([0, 2, 1])
    # # )
    #
    # print('rotation result here', batch_result.shape)
    #
    # batch_result = batch_result.reshape([bvh.frame_max_num, -1])
    # print(np.isclose(batch_result, joint_positions))
    #####################################################
    # About retarget between character in sketched
    # For sketched character retarget attribution setting
    # and let ground plane in the bottom of defined ground
    char_starting_location: List[float] = [0.0, 0.0, 0.0]
    # cache the starting worldspace location of character's root joint
    character_start_loc: npt.NDArray[np.float32] = np.array(char_starting_location, dtype=np.float32)

    # holds world coordinates of character root joint after retargeting
    char_root_positions: npt.NDArray[np.float32]

    ###############################################################################
    # get normal vector of projection plane
    # get every body group corresponding normal vector (In the process of animation, this will not change)
    joint_group_name_to_projection_plane: Dict[str, npt.NDArray[np.float32]] = {}
    # get every joint projection method vector (In the process of animation, this will not change)
    joint_to_projection_plane: Dict[str, npt.NDArray[np.float32]] = {}

    # bvh projection bodypart groups: each element in the list consists of
    # [group name : str, bvh_joint_names : List[str], projection method: sagittal ]
    bvh_projection_bodypart_groups: List[Tuple[str, List[str], str]] = \
        [
            ('Upper Limbs',
             ['RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'RightHandEnd', 'LeftShoulder',
              'LeftArm', 'LeftForeArm', 'LeftHand', 'LeftHandEnd'],
             'sagittal'),
            (
                'Lower Limbs',
                ['RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase', 'LeftUpLeg', 'LeftLeg',
                 'LeftFoot', 'LeftToeBase'],
                'pca'
            ),
            (
                'Trunk',
                ['Hips', 'Spine', 'Spine1', 'Spine2', 'Spine3', 'Neck', 'Head'],
                'frontal'
            )
        ]

    def find_suitable_projection_normal(group_name: str, joint_names: List[str], projection_method: str):
        # 1. decide projection method and get its corresponding normal vector
        # if we only choose frontal or saqittal,
        # it will not caculate correct normal vector
        if projection_method == 'frontal':
            return x_axis
        elif projection_method == 'sagittal':
            return z_axis
        elif projection_method == 'pca':
            pass
        else:
            assert NotImplementedError

        # 2. get joint projection group
        joints_idxs = [bvh_joint_names.index(joint_name) for joint_name in joint_names]
        # 3. To derive the joints name that demonstrate in current group
        # [T,V*C=3]
        joints_mask = np.full(joint_positions.shape[1], False, dtype=np.bool8)
        for idx in joints_idxs:
            joints_mask[3 * idx:3 * (idx + 1)] = True
        joints_points = joint_positions[:, joints_mask]

        # [T*selected_V,C=3]
        joints_points = joints_points.reshape([-1, 3])

        # get normal vector from PCA
        pca = PCA()
        pca.fit(joints_points)
        # get 3rd as the normal vector of projection
        pc3: npt.NDArray[np.float32] = pca.components_[2]

        # see if it is closer to the x axis or z axis
        x_cos_sim: float = np.dot(x_axis, pc3) / (np.linalg.norm(x_axis) * np.linalg.norm(pc3))
        z_cos_sim: float = np.dot(z_axis, pc3) / (np.linalg.norm(z_axis) * np.linalg.norm(pc3))

        # choose closer normal vector
        if abs(x_cos_sim) > abs(z_cos_sim):
            return x_axis
        else:
            return z_axis

    for joint_projection_group in bvh_projection_bodypart_groups:
        group_name = joint_projection_group[0]
        joint_names = joint_projection_group[1]
        projection_method = joint_projection_group[2]

        # for every joint group, consider and find its suitable projection plane normal
        projection_plane_vector = find_suitable_projection_normal(group_name, joint_names, projection_method)
        joint_group_name_to_projection_plane[group_name] = projection_plane_vector

        for joint_name in joint_names:
            joint_to_projection_plane[joint_name] = projection_plane_vector

    ###############################################################################
    # bvh depth derive into the animation 2D
    # [
    #  bvh_joint_name for depth : str ,
    #  char_joints group that acquire depth from given bvh joint name : List[str]
    #  ]
    char_bodypart_groups: List[Tuple[str, List[str]]] = \
        [
            ('Hips', ['right_shoulder', 'left_shoulder', 'right_hip',
                      'left_hip', 'hip', 'torso', 'neck']),
            ('LeftHand', ['left_elbow', 'left_hand']),
            ('RightHand', ['right_elbow', 'right_hand']),
            ('LeftFoot', ['left_knee', 'left_foot']),
            ('RightFoot', ['right_knee', 'right_foot'])
        ]

    # Here we redefine the depth determined by their projection vector
    # compute each bvh joint withn considered bvh projection mapping groups,
    # their distance to project
    bvh_joint_to_projection_depth: Dict[str, npt.NDArray[np.float32]] = {}

    def compute_depth_by_normal_vector() -> Dict[str, npt.NDArray[np.float32]]:
        bvh_joint_to_projection_depth: Dict[str, npt.NDArray[np.float32]] = {}

        for joint_name in bvh_joint_names:
            joint_idx = bvh_joint_names.index(joint_name)
            joint_xyz = joint_positions[:, 3 * joint_idx:3 * (joint_idx + 1)]
            projection_plane_normal_vector = joint_to_projection_plane[joint_name]

            # project bone onto 2D plane
            # if x axis is the suitable normal vector
            # depth = x
            if np.array_equal(projection_plane_normal_vector, x_axis):
                joint_depths = joint_xyz[:, 0]
            elif np.array_equal(projection_plane_normal_vector, z_axis):
                joint_depths = joint_xyz[:, 2]
            else:
                assert NotImplementedError, "Other projection way has not't been implemented!!"

            bvh_joint_to_projection_depth[joint_name] = joint_depths

        return bvh_joint_to_projection_depth

    bvh_joint_to_projection_depth = compute_depth_by_normal_vector()

    # This variable will not used
    char_joint_to_orientation: Dict[str, npt.NDArray[np.float32]] = {}


def test_vector_pendicular():
    vector1 = Vectors([1, 0, 0])
    vector2 = Vectors([2, 0, 0])
    l = []
    l.append(vector1)
    l.append(vector2)
    perpendicular_vector = Vectors(l).average().perpendicular()
    print(perpendicular_vector)


if __name__ == '__main__':
    test_bvh_retarget()
