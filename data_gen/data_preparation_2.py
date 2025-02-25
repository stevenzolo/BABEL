"""
Read pose/annotation training data, and extract action clip from continuous motions
"""
import os
import json
import pickle
import numpy as np
from scipy.interpolate import interp1d
from joblib import Parallel, delayed
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

    def attached_data(self, start_stamp, end_stamp, normal_fps=60):
        # first 22 joints correspond to body, the remained 30 ones belong to fingers
        npz_data = np.load(self.belong_to_mocap)
        ori_fps = int(npz_data['mocap_framerate'])
        # get 60 fps data
        ori_num_frames = len(npz_data['poses'])
        interp_func = interp1d(
            np.linspace(1, ori_num_frames, ori_num_frames), npz_data['poses'].astype(np.float32),
            axis=0, kind='nearest'
        )
        sampled_num_frms = int(ori_num_frames * (normal_fps / ori_fps))
        sampled_pose_data = interp_func(np.linspace(1, ori_num_frames, sampled_num_frms))
        return sampled_pose_data[int(normal_fps * start_stamp): int(normal_fps * end_stamp), :22 * 3]


def extract_movement_segments(data_local_dir, annotation_file_path, save_pkl_path=None):
    # Read annotation files and corresponding 'amass' pose data, extract and collect movement clips
    pose_attach_stats = []

    def _attach_single_file(_ann_amass_file):
        mocap_path = _ann_amass_file["feat_p"]
        if _ann_amass_file["frame_ann"] is None:  # example: 4887,
            return
        frame_ann_labels = _ann_amass_file["frame_ann"]["labels"]
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
                pass
        return

    movement_segment_collection = dict()
    with open(annotation_file_path, 'r') as btj:
        annotation_files = json.load(btj)

    # single_cpu_start = time.time()
    for amass_id in annotation_files:
        _attach_single_file(annotation_files[amass_id])
    # single_cpu_end = time.time()
    # print("Single CPU spends time of {}".format((single_cpu_end - single_cpu_start)/60))

    # mul_cpu_start = time.time()
    # Parallel(n_jobs=4)(delayed(_attach_single_file)(ann_amass) for ann_amass in annotation_files.values())
    # mul_cpu_end = time.time()
    # print("Multiple CPU spends time of {}".format((mul_cpu_end - mul_cpu_start) / 60))

    print("Attach {}/{} pose data of annotated segments".format(sum(pose_attach_stats), len(pose_attach_stats)))
    if save_pkl_path is not None:
        with open(save_pkl_path, 'wb') as sjp:
            pickle.dump(movement_segment_collection, sjp)
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
    annotation_train_path = os.path.join(os.getcwd(), 'data', 'babel_v1.0_release', 'train.json')
    movement_segment_dict = extract_movement_segments(
        data_local_dir=local_dir,
        annotation_file_path=annotation_train_path
    )
    visualize_angle_arr_in_movements(movement_segment_dict)

