"""
Read pose/annotation training data, and extract action clip from continuous motions
"""
import os
import json
import numpy as np
from matplotlib import pyplot as plt

from utils import JOINT_ANGLE_ORDER_LST
import matplotlib
matplotlib.use('TkAgg')


class MovementSegment:
    def __init__(self, belong_to_mocap, start_stamp, end_stamp, movement_labels):
        self.belong_to_mocap = belong_to_mocap
        self.labels = movement_labels
        if os.path.exists(belong_to_mocap):
            self.pose = self.attached_data(start_stamp, end_stamp)
        else:
            self.pose = None

    def attached_data(self, start_stamp, end_stamp):
        # first 22 joints correspond to body, the remained 30 ones belong to fingers
        npz_data = np.load(self.belong_to_mocap)
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
        return data_pose[int(60 * start_stamp): int(60 * end_stamp), :22 * 3]  # first 22 joints belong to body


def extract_movement_segments(data_local_dir, annotation_file_path):
    # Read annotation files and corresponding 'amass' pose data, extract and collect movement clips
    movement_segment_collection = dict()
    with open(annotation_file_path, 'r') as btj:
        annotation_train = json.load(btj)
    pose_attach_stats = []
    for amass_id in annotation_train:
        mocap_path = annotation_train[amass_id]["feat_p"]
        if annotation_train[amass_id]["frame_ann"] is None:  # example: 4887,
            continue
        frame_ann_labels = annotation_train[amass_id]["frame_ann"]["labels"]
        for label in frame_ann_labels:
            movement_segment = MovementSegment(
                    belong_to_mocap=os.path.join(data_local_dir, mocap_path),
                    start_stamp=label["start_t"],
                    end_stamp=label["end_t"],
                    movement_labels=label["act_cat"]
                )
            if movement_segment.pose is not None:
                for movement_lbl in movement_segment.labels:
                    if movement_lbl in movement_segment_collection:
                        movement_segment_collection[movement_lbl].append(movement_segment)
                    else:
                        movement_segment_collection[movement_lbl] = [movement_segment]
                pose_attach_stats.append(1)
            else:
                pose_attach_stats.append(0)

    print("Attach {}/{} pose data of annotated segments".format(sum(pose_attach_stats), len(pose_attach_stats)))
    return movement_segment_collection


def visualize_angle_arr_in_movements(movement_segments_collection: dict):
    def _sample_normalization(arr_lst, scale_tole=5):
        min_len = min([len(arr) for arr in arr_lst])
        max_len = max([len(arr) for arr in arr_lst])
        if max_len > min_len * scale_tole:
            print("Maybe invalid sample normalization")
        sampled_lst = []
        for arr in arr_lst:
            sampled_arr = []
            for ix in range(min_len):
                sampled_arr.append(arr[min(len(arr)-1, int(ix * len(arr) / min_len))])
            sampled_lst.append(np.array(sampled_arr))
        return sampled_lst

    angle_arr_in_movements = dict()
    for joint_angle in JOINT_ANGLE_ORDER_LST:
        angle_arr_in_movements[joint_angle.name] = dict()
    for movement_lbl in movement_segments_collection:
        if movement_lbl == 'transition':
            continue
        for angle_name_ in angle_arr_in_movements:
            angle_arr_in_movements[angle_name_][movement_lbl] = []
        for movement in movement_segments_collection[movement_lbl]:
            for angle_ix, joint_angle in enumerate(JOINT_ANGLE_ORDER_LST):
                angle_arr_in_movements[joint_angle.name][movement_lbl].append(movement.pose[:, angle_ix])
    for angle_name_ in angle_arr_in_movements:
        plt.figure()
        for angle_in_move in angle_arr_in_movements[angle_name_].values():
            # one movement of varied humans may have varied length
            normalized_angle_arrs = _sample_normalization(angle_in_move)
            mean_arr = np.mean(normalized_angle_arrs, axis=0)
            var_arr = np.var(normalized_angle_arrs, axis=0)
            plt.fill_between(np.arange(len(mean_arr)), mean_arr-var_arr, mean_arr+var_arr)
        plt.show()
        plt.close()
    return


if __name__ == "__main__":
    local_dir = os.path.join(os.getcwd(), 'AMASS_Data')
    annotation_train_path = os.path.join(os.getcwd(), 'babel_v1.0_release', 'train.json')
    movement_segment_dict = extract_movement_segments(
        data_local_dir=local_dir,
        annotation_file_path=annotation_train_path
    )
    visualize_angle_arr_in_movements(movement_segment_dict)

