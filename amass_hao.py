import numpy as np
from enum import IntEnum
from mpl_toolkits.mplot3d import Axes3D
from transforms3d import quaternions as quat
from numpy.linalg import norm
from articulate.armature import Bone, Joint
from kinematics.skeleton_kinematics import SkeletonDimensions
from kinematics.contact_judgment import contact_judgment, contact_judgment_postprocess
from kinematics.skeleton_kinematics_smpl import Skeleton
from config import *
from utils import *

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class AMASSJoint(IntEnum):
    ROOT = 0
    LEFT_HIP = 1
    RIGHT_HIP = 2
    WAIST_JOINT = 3
    LEFT_KNEE = 4
    RIGHT_KNEE = 5
    SPINE_JOINT = 6
    LEFT_ANKLE = 7
    RIGHT_ANKLE = 8
    SPINE1_JOINT = 9
    LEFT_TOE_JOINT = 10
    RIGHT_TOE_JOINT = 11
    NECK_JOINT = 12
    LEFT_CLAVICLE = 13
    RIGHT_CLAVICLE = 14
    HEAD_JOINT = 15
    LEFT_SHOULDER_JOINT = 16
    RIGHT_SHOULDER_JOINT = 17
    LEFT_ELBOW = 18
    RIGHT_ELBOW = 19
    LEFT_WRIST = 20
    RIGHT_WRIST = 21
    SPINE2_JOINT = 22 # dummy joint to match with current skeleton
    AMASS_JOINT_NUM = 23


amass_joint_proximal_bone_list = [None, Bone.PELVIS,     Bone.PELVIS,      Bone.PELVIS, Bone.LEFT_THIGH, Bone.RIGHT_THIGH,
                                  Bone.SPINE, Bone.LEFT_SHANK, Bone.RIGHT_SHANK, Bone.SPINE1, Bone.LEFT_FOOT, Bone.RIGHT_FOOT,
                                  Bone.NECK, Bone.SPINE2, Bone.SPINE2, Bone.NECK1,
                                  Bone.LEFT_SHOULDER, Bone.RIGHT_SHOULDER, Bone.LEFT_ARM, Bone.RIGHT_ARM,
                                  Bone.LEFT_FOREARM, Bone.RIGHT_FOREARM, Bone.SPINE2]
amass_joint_distal_bone_list =   [None, Bone.LEFT_THIGH, Bone.RIGHT_THIGH, Bone.SPINE,  Bone.LEFT_SHANK, Bone.RIGHT_SHANK,
                                  Bone.SPINE1, Bone.LEFT_FOOT, Bone.RIGHT_FOOT, Bone.SPINE2, None, None,
                                  Bone.NECK1, Bone.LEFT_SHOULDER, Bone.RIGHT_SHOULDER, Bone.HEAD,
                                  Bone.LEFT_ARM, Bone.RIGHT_ARM, Bone.LEFT_FOREARM, Bone.RIGHT_FOREARM,
                                  Bone.LEFT_HAND, Bone.RIGHT_HAND, Bone.NECK]

amass_joint_posi_order_list = [AMASSJoint.LEFT_HIP, AMASSJoint.RIGHT_HIP, AMASSJoint.WAIST_JOINT,
                               AMASSJoint.LEFT_KNEE, AMASSJoint.RIGHT_KNEE, AMASSJoint.SPINE_JOINT, AMASSJoint.LEFT_ANKLE,
                               AMASSJoint.RIGHT_ANKLE, AMASSJoint.SPINE1_JOINT, AMASSJoint.LEFT_TOE_JOINT, AMASSJoint.RIGHT_TOE_JOINT,
                               AMASSJoint.SPINE2_JOINT, AMASSJoint.LEFT_CLAVICLE, AMASSJoint.RIGHT_CLAVICLE, AMASSJoint.NECK_JOINT,
                               AMASSJoint.HEAD_JOINT, AMASSJoint.LEFT_SHOULDER_JOINT, AMASSJoint.RIGHT_SHOULDER_JOINT,
                               AMASSJoint.LEFT_ELBOW, AMASSJoint.RIGHT_ELBOW, AMASSJoint.LEFT_WRIST, AMASSJoint.RIGHT_WRIST]


def read_amass_data_file(file_name):
    '''
    For data preprocessing of reference data .pt
    '''
    print(file_name)
    pose_data_file = []
    try:
        data = np.load(file_name)
    except:
        return None

    framerate = int(data['mocap_framerate'])
    # get 60 fps data
    if framerate == 120:
        step = 2
    elif framerate == 60 or framerate == 59:
        step = 1
    else:
        return None

    data_pose = data['poses'][::step].astype(np.float32)
    data_trans = data['trans'][::step].astype(np.float32)
    data_beta = data['betas'][:10]
    length = data['poses'][::step].shape[0]
    length = torch.tensor(length, dtype=torch.int)
    shape = torch.tensor(np.asarray(data_beta, np.float32))
    tran = torch.tensor(np.asarray(data_trans, np.float32))
    pose = torch.tensor(np.asarray(data_pose, np.float32)).view(-1, 52, 3)
    pose[:, 23] = pose[:, 37]  # right hand
    pose = pose[:, :24].clone()  # only use body

    skeleton = Skeleton(pose, tran, shape, paths.smpl_file)
    skeleton.view()

    file_bone_orientation_data, file_bone_position_data, file_joint_position_data, file_joint_orientation_data = pose_reconstruction(data_pose, data_trans)

    file_bone_velocity_data = synthesize_vel_data(file_bone_position_data)
    file_bone_acc_data = synthesize_acc_data(file_bone_position_data)
    file_bone_angular_vel_data = synthesize_angular_vel_data(file_bone_orientation_data)

    for bone_ori, bone_pos,  bone_vel, bone_acc, bone_angular_vel, joint_ori, joint_pos in zip(file_bone_orientation_data,file_bone_position_data,
                                               file_bone_velocity_data, file_bone_acc_data,
                                               file_bone_angular_vel_data, file_joint_orientation_data, file_joint_position_data):
        p = []
        for b_o, b_p, b_v, b_a, b_av, j_o, j_p in zip(bone_ori, bone_pos,  bone_vel, bone_acc, bone_angular_vel, joint_ori, joint_pos):
            array = list(b_o)
            array.extend(b_p)
            array.extend(b_v)
            array.extend(b_a)
            array.extend(b_av)
            array.append(False)  # calculate later
            array.extend(j_o)
            array.extend(j_p)
            p.append(array)
        pose_data_file.append(p)

    pose_data_file = np.array(pose_data_file)
    return pose_data_file


def read_amass_data_files(files):
    pose_data_list = []
    for file_name in files:
        pose_data_file = read_amass_data_file(file_name)
        pose_data_list.append(pose_data_file)
    return pose_data_list


def pose_reconstruction(posture_data, trans_data, skeleton=None, plot_flag=False):
    '''
    For data preprocessing of reference data .pt
    '''
    bone_ori_data = []
    bone_pos_data = []
    joint_pos_data = []
    joint_ori_data = []

    if skeleton is None:
        skeleton = SkeletonDimensions()
    for posture_list, root_pos in zip(posture_data, trans_data):
        bone_quat_list = []
        relative_quat_list = []
        bone_posi_list = []
        joint_posi_list = []  # bone's proximal joint position
        joint_ori_list = []
        # init
        for amass_joint_index in range(AMASSJoint.AMASS_JOINT_NUM):
            aa = posture_list[amass_joint_index * 3 : amass_joint_index * 3 + 3]
            theta = norm(aa)
            if theta > 0.:
                vec = [v / theta for v in aa]
            else:
                vec = [1., 0., 0.]
            relative_quat_list.append(quat.axangle2quat(vec, theta))
        for bone_index in range(Bone.BONE_NUM):
            bone_posi_list.append([0., 0., 0.])
            joint_posi_list.append([0., 0., 0.])
            bone_quat_list.append([1., 0., 0., 0.])
            joint_ori_list.append([1., 0., 0., 0.])
        bone_posi_list[Bone.PELVIS] = np.dot(root_pos, np.array([[1,0,0],[0,0,-1],[0,1,0]]))
        joint_posi_list[Bone.PELVIS] = np.dot(root_pos, np.array([[1,0,0],[0,0,-1],[0,1,0]]))
        joint_ori_list[Joint.ROOT] = bone_quat_list[Bone.PELVIS] = quat.qmult(quat.mat2quat(np.array([[1,0,0],[0,0,1],[0,-1,0]])),relative_quat_list[0])
        # bone_posi_list[Bone.PELVIS] = root_pos
        # joint_posi_list[Bone.PELVIS] = root_pos
        # bone_quat_list[Bone.PELVIS] = relative_quat_list[0]

        # iteration from proximal to distal
        for amass_joint_index in amass_joint_posi_order_list:
            proximal_bone_index = amass_joint_proximal_bone_list[amass_joint_index]
            distal_bone_index = amass_joint_distal_bone_list[amass_joint_index]
            if distal_bone_index is None:
                continue
            # orientation
            if amass_joint_index != AMASSJoint.SPINE2_JOINT:
                bone_quat_list[distal_bone_index] = quat.qmult(bone_quat_list[proximal_bone_index],relative_quat_list[amass_joint_index])
                joint_ori_list[distal_bone_index] = relative_quat_list[amass_joint_index]
            else:
                bone_quat_list[distal_bone_index] = bone_quat_list[proximal_bone_index]
                joint_ori_list[distal_bone_index] = np.array([1,0,0,0])
            # position
            if proximal_bone_index == Bone.PELVIS:
                if distal_bone_index == Bone.RIGHT_THIGH:
                    lower_vec_bone_index = 21
                elif distal_bone_index == Bone.LEFT_THIGH:
                    lower_vec_bone_index = 22
                else:
                    lower_vec_bone_index = proximal_bone_index
            elif proximal_bone_index == Bone.SPINE2:
                if distal_bone_index == Bone.RIGHT_SHOULDER:
                    lower_vec_bone_index = 23
                elif distal_bone_index == Bone.LEFT_SHOULDER:
                    lower_vec_bone_index = 24
                else:
                    lower_vec_bone_index = proximal_bone_index
            else:
                lower_vec_bone_index = proximal_bone_index
            upper_vec_bone_index = distal_bone_index
            bone_posi_list[distal_bone_index] = bone_posi_list[proximal_bone_index] \
                                           + quat.rotate_vector(skeleton.bone_lower_vec[lower_vec_bone_index],
                                                                bone_quat_list[proximal_bone_index]) \
                                           - quat.rotate_vector(skeleton.bone_upper_vec[upper_vec_bone_index],
                                                                bone_quat_list[distal_bone_index])
            joint_posi_list[distal_bone_index] = bone_posi_list[proximal_bone_index] \
                                           + quat.rotate_vector(skeleton.bone_lower_vec[lower_vec_bone_index],
                                                                bone_quat_list[proximal_bone_index])

        # bone_posi_list2, joint_posi_list2 = sk.calculate_skeleton_bone_and_joint_position(skeleton,
        #                                                                                   bone_posi_list[Bone.PELVIS],
        #                                                                                   bone_quat_list)

        bone_ori_data.append(bone_quat_list)
        bone_pos_data.append(bone_posi_list)
        joint_pos_data.append(joint_posi_list)
        joint_ori_data.append(joint_ori_list)

        if plot_flag:
            fig = plt.figure()
            t=20
            ax = fig.add_subplot(projection='3d')
            ax = Axes3D(fig)
            x_bone = []
            y_bone = []
            z_bone = []
            for posi in joint_posi_list:
                x_bone.append(posi[0])
                y_bone.append(posi[1])
                z_bone.append(posi[2])
            ax.scatter(x_bone, y_bone, z_bone, c='y')
            max_range = max([max(x_bone) - min(x_bone), max(y_bone) - min(y_bone), max(z_bone) - min(z_bone)])
            Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (max(x_bone) + min(x_bone))
            Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (max(y_bone) + min(y_bone))
            Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (max(z_bone) + min(z_bone))
            # Comment or uncomment following both lines to test the fake bounding box:
            for xb, yb, zb in zip(Xb, Yb, Zb):
                ax.plot([xb], [yb], [zb], 'w')
            plt.show()

    return bone_ori_data, bone_pos_data, joint_pos_data, joint_ori_data


def obtain_root_pos_worldcoor_data(pose_data):
    '''
    For data preprocessing of reference data .pt
    '''
    frames = pose_data.shape[0]
    root_pos_data = []
    for frame_index in range(frames):
        root_pos_worldcor_list = []
        pos = pose_data[frame_index, 0, 4:7]
        root_pos_worldcor_list.append(pos)
        root_pos_data.append(root_pos_worldcor_list)
    return np.array(root_pos_data)


def obtain_bone_ori_worldcoor_data(pose_data, ori_format, bone_index_list=None):
    '''
    For data preprocessing of reference data .pt
    '''
    frames = pose_data.shape[0]
    bone_ori_data = []
    if bone_index_list is None:
        bone_index_list = list(range(Bone.BONE_NUM))
    for frame_index in range(frames):
        bone_ori_worldcor_list = []
        for bone_index in bone_index_list:
            # bone_orientation
            q_W_B = pose_data[frame_index, bone_index, 0:4]
            q_W_B = qw_sign_check(q_W_B)
            bone_ori = feature_format_conversion(q_W_B, QUATERNION, ori_format)
            bone_ori_worldcor_list.append(bone_ori)
        bone_ori_data.append(bone_ori_worldcor_list)
    return np.array(bone_ori_data)


def obtain_bone_acc_worldcoor_data(pose_data, bone_index_list=None):
    '''
    For data preprocessing of reference data .pt
    '''
    frames = pose_data.shape[0]
    bone_acc_data = []
    if bone_index_list is None:
        bone_index_list = list(range(Bone.BONE_NUM))
    for frame_index in range(frames):
        bone_acc_worldcor_list = []
        for bone_index in bone_index_list:
            # bone acceleration
            a_W_B = pose_data[frame_index, bone_index, 10:13]
            bone_acc_worldcor_list.append(a_W_B)
        bone_acc_data.append(bone_acc_worldcor_list)
    return np.array(bone_acc_data)


def obtain_joint_ori_data(pose_data, ori_format, joint_index_list=None):
    '''
    For data preprocessing of reference data .pt
    '''
    frames = pose_data.shape[0]
    joint_ori_data = []
    if joint_index_list is None:
        joint_index_list = list(range(Joint.JOINT_NUM))
    for frame_index in range(frames):
        joint_ori_list = []
        for joint_index in joint_index_list:
            # joint orientation
            q = pose_data[frame_index, joint_index, 17:21]
            q = qw_sign_check(q)
            joint_ori = feature_format_conversion(q, QUATERNION, ori_format)
            joint_ori_list.append(joint_ori)
        joint_ori_data.append(joint_ori_list)
    return np.array(joint_ori_data)


def obtain_amass_kinematic_single_data(amass_pose_data_file):
    data_file = {
        FULL_BONE_ORI_WORLD: [],
        FULL_BONE_POS_WORLD: [],
        FULL_BONE_VEL_WORLD: [],
        FULL_BONE_ACC_WORLD: [],
        FULL_BONE_ANGULAR_VEL_WORLD: [],
        FULL_BONE_CONTACT: [],
        JOINT_ORI_LOCAL: [],
        JOINT_POS_WORLD: [],
    }

    frames = amass_pose_data_file.shape[0]
    for frame_index in range(frames):
        bone_ori_worldcor_list = []
        bone_pos_worldcor_list = []
        bone_vel_worldcor_list = []
        bone_acc_worldcor_list = []
        bone_angular_vel_worldcor_list = []
        joint_ori_list = []
        joint_pos_worldcor_list = []
        for bone_index in range(Bone.BONE_NUM):
            # bone_orientation
            q_W_B = amass_pose_data_file[frame_index, bone_index, 0:4]
            if q_W_B[0] < 0.0:
                q_W_B = -q_W_B
            bone_ori_worldcor_list.append(q_W_B)

            # bone position
            x_W = amass_pose_data_file[frame_index, bone_index, 4:7]
            bone_pos_worldcor_list.append(x_W)

            # bone vel
            v_W = amass_pose_data_file[frame_index, bone_index, 7:10]
            bone_vel_worldcor_list.append(v_W)

            # bone acceleration
            a_W_B = amass_pose_data_file[frame_index, bone_index, 10:13]  # TODO: need * GRAVITY ?
            bone_acc_worldcor_list.append(a_W_B)

            # bone angular vel
            angular_vel_W_B = amass_pose_data_file[frame_index, bone_index, 13:16]
            bone_angular_vel_worldcor_list.append(angular_vel_W_B)

            # joint orientation
            q_W_B = amass_pose_data_file[frame_index, bone_index, 17:21]
            if q_W_B[0] < 0.0:
                q_W_B = -q_W_B
            joint_ori_list.append(q_W_B)

            # joint position
            x_W = amass_pose_data_file[frame_index, bone_index, 21:24]
            joint_pos_worldcor_list.append(x_W)

        data_file[FULL_BONE_ORI_WORLD].append(bone_ori_worldcor_list)
        data_file[FULL_BONE_POS_WORLD].append(bone_pos_worldcor_list)
        data_file[FULL_BONE_VEL_WORLD].append(bone_vel_worldcor_list)
        data_file[FULL_BONE_ACC_WORLD].append(bone_acc_worldcor_list)
        data_file[FULL_BONE_ANGULAR_VEL_WORLD].append(bone_angular_vel_worldcor_list)
        data_file[JOINT_ORI_LOCAL].append(joint_ori_list)
        data_file[JOINT_POS_WORLD].append(joint_pos_worldcor_list)

    # post-process contact judgment for AMASS_Data dataset
    bone_contact_data = np.array(contact_judgment_postprocess(amass_pose_data_file[..., 21:24]))
    bone_contact_data = bone_contact_data.reshape(-1, 21, 1)
    data_file[FULL_BONE_CONTACT] = bone_contact_data

    return data_file


def obtain_amass_kinematic_data(amass_pose_data_list):
    data = []
    for amass_pose_data_file in amass_pose_data_list:
        data_file = obtain_amass_kinematic_single_data(amass_pose_data_file)
        data.append(data_file)
    return data


def load_kinematic_single_data(file_name):
    pose_data = read_amass_data_file(file_name)
    data = obtain_amass_kinematic_single_data(pose_data)
    return data


def load_kinematic_data(file_list):
    pose_data_list = read_amass_data_files(file_list)
    data_list = obtain_amass_kinematic_data(pose_data_list)
    return data_list
