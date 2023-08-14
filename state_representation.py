"""
Scan the train action clips and segment with rdp parameter (trialed), then
    cluster the segmented angle range (cluster distance).
Thus, an action range of each angle is assigned with trained cluster number,
    continuous movements -> cluster no. combine
"""
import os
import numpy as np
from rdp import rdp
from copy import deepcopy
from transforms3d import euler
from matplotlib import pyplot as plt

from preprocess import select_related_angles
from data_preparation import extract_movement_segments
from utils import JointAngle, JOINT_ANGLE_ORDER_LST, ANGLE_CLUSTER_ORDER_LST
import matplotlib
matplotlib.use('TkAgg')


class AmassClip:
    def __init__(self, data_path, angle_order_lst: [JointAngle]):
        self.angle_order_lst = angle_order_lst
        self.file_name = '/'.join(data_path.split('/')[-2:])   # file and its folder
        data_pose = self.load_data(data_path, body_pose_bound=len(angle_order_lst))
        for frame_pose in data_pose:
            for angle_ix in range(int(len(angle_order_lst)/3)):
                joint_relative_quat = JointAngle.axangle2quat(frame_pose[angle_ix * 3: angle_ix * 3 + 3])
                rotation, extension, abduction = euler.quat2euler(joint_relative_quat, axes='rxzy')
                self.angle_order_lst[angle_ix * 3].append_values_in_clip(rotation)
                self.angle_order_lst[angle_ix * 3 + 1].append_values_in_clip(extension)
                self.angle_order_lst[angle_ix * 3 + 2].append_values_in_clip(abduction)
        self.correct_jump_angle_vals()

    def load_data(self, data_path, body_pose_bound):
        # first 22 joints correspond to body, the remained 30 ones belong to fingers
        npz_data = np.load(data_path)
        framerate = int(npz_data['mocap_framerate'])
        # get 60 fps data
        if framerate == 120:
            step = 2
        elif framerate == 60 or framerate == 59:
            step = 1
        else:
            raise ValueError("Undefined frame rate")
        data_pose = npz_data['poses'][::step].astype(np.float32)
        # data_pose = npz_data['poses'][:10].astype(np.float32)   # for test
        # data_trans = npz_data['trans'][::step].astype(np.float32)
        return data_pose[:, :body_pose_bound]

    def correct_jump_angle_vals(self):
        """
        correct angle jump like -3.14 --> 3.14
        """
        jump_angle_lst = []
        for angle in self.angle_order_lst:
            angle_vals = angle.value_arr_in_clip
            for frm_ix in range(1, len(angle_vals)):
                if angle_vals[frm_ix] * angle_vals[frm_ix - 1] < 0 and abs(
                        angle_vals[frm_ix] - angle_vals[frm_ix - 1]) > np.pi:
                    print("jump value appears at {} in {}".format(angle.name, self.file_name))
                    jump_angle_lst.append(angle)
                    break

        for jump_angle in jump_angle_lst:  # todo, for one angle, jump correction in one clip affects the whole cluster?
            corrected_val_frms = []
            for val in jump_angle.value_arr_in_clip:
                corrected_val_frms.append(val - np.pi * 0.5) if val > 0 else corrected_val_frms.append(
                    val + np.pi * 1.5)
            jump_angle.value_arr_in_clip = corrected_val_frms
        return

    def visualize_segmentation(self):
        # visualize segmentation of angle trends in one file
        all_segments_stamp = []
        for angle in self.angle_order_lst:
            all_segments_stamp.extend(angle.segment_stamps_in_clip)
        all_segments_stamp = sorted(list(set(all_segments_stamp)))

        for angle in self.angle_order_lst:
            if max(angle.value_arr_in_clip) - min(angle.value_arr_in_clip) > 0.5:
                plt.plot(angle.value_arr_in_clip)  # only plot angle array with big change
        for stamp in all_segments_stamp:
            plt.plot([stamp, stamp], [-2.0, 2.0], 'y')
        plt.show()

    def segment_joint_angle_frms(self, rdp_epsilon=0.3):
        # todo, adaptive epsilon
        for angle in self.angle_order_lst:
            segment_lst = rdp(
                np.array([(ix, val) for ix, val in enumerate(angle.value_arr_in_clip)]),
                epsilon=rdp_epsilon
            )
            last_val = None
            for stamp_ix, (frm_ix, val) in enumerate(segment_lst):
                angle.segment_stamps_in_clip.append(int(frm_ix))
                if not stamp_ix == 0:
                    angle.segment_value_in_clip.append(val - last_val)
                last_val = val
        return

    def reset_stamp_match(self):
        for angle in self.angle_order_lst:
            angle.stamp_match_start_ix = 0


def detect_angle_changes_over_train(
        movement_segments_collection: dict, selected_feature_idx: list, undetected_change=0.1
):
    """
    calculate rdp_epsilon for each joint angle, for that some angle has big range while others not
    """
    angle_changes_collection = dict()
    for movement_label in movement_segments_collection:
        angle_changes_lst_in_movement = [[] for _ in range(len(selected_feature_idx))]
        for movement in movement_segments_collection[movement_label]:
            for idx, angle_ix in enumerate(selected_feature_idx):
                angle_val_change = movement.pose[:, angle_ix].max() - movement.pose[:, angle_ix].min()
                angle_changes_lst_in_movement[idx].append(angle_val_change)
        # angle_changes_mean = [np.mean(angle_changes_lst) for angle_changes_lst in angle_changes_lst_in_movement]

        for idx, angle_ix in enumerate(selected_feature_idx):
            # all people have big change in this angle when doing movement
            min_change = min(angle_changes_lst_in_movement[idx])
            if JOINT_ANGLE_ORDER_LST[angle_ix].name in angle_changes_collection:
                angle_changes_collection[JOINT_ANGLE_ORDER_LST[angle_ix].name].append(min_change)
            else:
                angle_changes_collection[JOINT_ANGLE_ORDER_LST[angle_ix].name] = [min_change]

    min_angle_change_detection = {}
    for angle_name_, changes_in_movements in angle_changes_collection.items():
        # detect possible changes in all related movements
        min_angle_change_detection[angle_name_] = min(changes_in_movements)
    return min_angle_change_detection   # all zero in selected 11 angle features


def cluster_angles_over_train_files(data_local_dir, rdp_epsilon):
    amass_clip_lst = []
    amass_mocap_path_lst = []
    for cur_path, directories, file_names in os.walk(os.path.join(data_local_dir, 'AMASS_Data')):
        for file_name in file_names:
            if file_name.split('.')[-1] == 'npz':
                amass_mocap_path_lst.append(os.path.join(cur_path, file_name))

    for file_path in amass_mocap_path_lst:
        amass_clip = AmassClip(data_path=file_path, angle_order_lst=deepcopy(JOINT_ANGLE_ORDER_LST))
        amass_clip.segment_joint_angle_frms()   # todo, better epsilon
        # amass_file.visualize_segmentation()
        amass_clip_lst.append(amass_clip)
        for angle_ix, angle in enumerate(amass_clip.angle_order_lst):
            ANGLE_CLUSTER_ORDER_LST[angle_ix].add_joint_segments(angle.segment_value_in_clip)

    # after collecting segments of all training files, cluster the segments
    for joint_cluster in ANGLE_CLUSTER_ORDER_LST:
        joint_cluster.cluster_training()
    # represent single file as discrete cluster of angles, generate training data for RL
    for amass_clip in amass_clip_lst:
        for angle, angle_cluster in zip(amass_clip.angle_order_lst, ANGLE_CLUSTER_ORDER_LST):
            for segment_val in angle.segment_value_in_clip:
                angle.cluster_no_in_clip.append(int(angle_cluster.predict(segment_val)))
    return amass_clip_lst


def generate_cluster_primitives_data(amass_clip_lst: [AmassClip]):
    primitive_state_collection = []
    movement_collection = []    # todo: correspond to primitive list and duration
    for amass_clip in amass_clip_lst:
        all_stamps_lst = []
        for angle in amass_clip.angle_order_lst:
            all_stamps_lst.extend(angle.segment_stamps_in_clip)
        sorted_stamps = sorted(list(set(all_stamps_lst)))
        amass_clip.reset_stamp_match()

        for clip_stamp in sorted_stamps[:-1]:   # except the last frame
            combine_cluster_no_lst = []
            for angle in amass_clip.angle_order_lst:
                angle_stamps = angle.segment_stamps_in_clip
                for stamp_ix in range(angle.stamp_match_start_ix, len(angle_stamps)-1):
                    if angle_stamps[stamp_ix + 1] > clip_stamp >= angle_stamps[stamp_ix]:
                        combine_cluster_no_lst.append(repr(angle.cluster_no_in_clip[stamp_ix]))
                        angle.stamp_match_start_ix = stamp_ix
                        break
            cluster_no_combination = '_'.join(combine_cluster_no_lst)
            primitive_state_collection.append(cluster_no_combination)
    return


if __name__ == "__main__":
    local_dir = os.path.join(os.getcwd(), 'AMASS_Data')
    annotation_train_path = os.path.join(os.getcwd(), 'babel_v1.0_release', 'train.json')
    movement_segment_dict = extract_movement_segments(
        data_local_dir=local_dir,
        annotation_file_path=annotation_train_path
    )
    optimal_feature_set = select_related_angles(movement_segment_dict)
    joint_angle_name_order_lst = [joint.name for joint in JOINT_ANGLE_ORDER_LST]
    optimal_feature_idx_lst = [joint_angle_name_order_lst.index(feature) for feature in optimal_feature_set]
    angle_clip_base = detect_angle_changes_over_train(movement_segment_dict, optimal_feature_idx_lst)
    cluster_angles_over_train_files(local_dir, angle_clip_base)
