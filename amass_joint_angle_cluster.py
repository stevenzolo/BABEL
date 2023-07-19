"""
Given the relative pose of joint, regardless of the order, the joint angle can be calculated.
Plot the joint angle trending and try to cluster.

Author@Weichao
Created on Jul. 8th, 2023
"""
import os
import json
import numpy as np
from copy import deepcopy
from numpy.linalg import norm
from transforms3d import quaternions as quat
from transforms3d import euler
from rdp import rdp
from sklearn.cluster import AgglomerativeClustering, KMeans
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


class JointAngle:
    def __init__(self, name):
        self.name = name
        self.segment_stamps_in_clip = []   # start from 0 and end at total clip frames
        self.segment_value_in_clip = []  # value change between stamps
        self.cluster_no_in_clip = []   # one clip with N clips has N-1 segment/cluster.
        self.value_arr_in_clip = []
        self.stamp_match_start_ix = 0

    def append_values_in_clip(self, frame_val):
        self.value_arr_in_clip.append(frame_val)

    @staticmethod
    def axangle2quat(axangle_arr):
        angle_norm = norm(axangle_arr)
        if angle_norm > 0.0:
            vec = axangle_arr / angle_norm
            return quat.axangle2quat(vec, angle_norm)
        else:
            return np.array([1., 0., 0., 0.])


class MovementSegment:
    def __init__(self, belong_to_mocap, start_stamp, end_stamp, movement_label):
        self.belong_to_mocap = belong_to_mocap
        self.label = movement_label
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


class AngleCluster:
    """
    Initialize with joint name and their segments range.
    Then cluster based one all train files. i.e. a training process
    Output: given segment range of specific joint, output the belonged cluster.

    todo: cluster of one joint --> cluster joints together for more compatible state distinguishing.
        (adaptive cluster parameter for joints)
    """
    def __init__(self, angle_name):
        self.angle_name = angle_name
        self.segments_collection = []
        self.kmeans_cluster = None

    def add_joint_segments(self, segment_one_file):
        self.segments_collection.extend(segment_one_file)
        return

    def cluster_training(self, agg_dist_thresh=0.3):
        """
        cluster and representation, todo: select one cluster method over many ones
        challenge & requirements:
            - n_cluster is unknown priorly, which cannot be given when clustering;
            - the cluster method should have 'predict' method (cluster can be generalized);
            distance (range) has varied sensitivity to joints, e.g. hip v.s. ankle
        """
        segment_range_arr = np.array([(0, x) for x in self.segments_collection])  # expand data into 2D
        agg_clustering = AgglomerativeClustering(
            distance_threshold=agg_dist_thresh,  # smaller threshold produces more clusters
            n_clusters=None
        ).fit(segment_range_arr)
        kmeans_clustering = KMeans(n_clusters=agg_clustering.n_clusters_).fit(segment_range_arr)
        self.kmeans_cluster = kmeans_clustering

    def predict(self, seg_val):
        return self.kmeans_cluster.predict(np.array([(0, seg_val)]))

    def __print__(self):
        return self.angle_name


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
            if len(label["act_cat"]) == 1:
                movement_label = label["act_cat"][0]
            else:
                movement_label = '_'.join(label["act_cat"])
            movement_segment = MovementSegment(
                    belong_to_mocap=os.path.join(data_local_dir, mocap_path),
                    start_stamp=label["start_t"],
                    end_stamp=label["end_t"],
                    movement_label=label["act_cat"]
                )
            if movement_segment.pose is not None:
                if movement_label in movement_segment_collection:
                    movement_segment_collection[movement_label].append(movement_segment)
                else:
                    movement_segment_collection[movement_label] = [movement_segment]
                pose_attach_stats.append(1)
            else:
                pose_attach_stats.append(0)

    print("Attach {}/{} pose data of annotated segments".format(sum(pose_attach_stats), len(pose_attach_stats)))
    return movement_segment_collection


def detect_angle_changes_over_train(movement_segments_collection: dict, undetected_change=0.1):
    angle_changes_collection = dict()
    for movement_label in movement_segments_collection:
        angle_changes_lst_in_movement = [[] for _ in range(len(joint_angle_order_lst))]
        for movement in movement_segments_collection[movement_label]:
            for angle_ix in range(len(joint_angle_order_lst)):
                angle_val_change = movement.pose[:, angle_ix].max() - movement.pose[:, angle_ix].min()
                angle_changes_lst_in_movement[angle_ix].append(angle_val_change)
        angle_changes_mean = [np.mean(angle_changes_lst) for angle_changes_lst in angle_changes_lst_in_movement]

        for angle_ix, joint_angle in enumerate(joint_angle_order_lst):
            if angle_changes_mean[angle_ix] > undetected_change:  # todo, plus variance?
                # all people have big change in this angle when doing movement
                min_change = min(angle_changes_lst_in_movement[angle_ix])
                if joint_angle.name in angle_changes_collection:
                    angle_changes_collection[joint_angle.name].append(min_change)
                else:
                    angle_changes_collection[joint_angle.name] = [min_change]

    min_angle_change_detection = {}
    for angle_name_, changes_in_movements in angle_changes_collection.items():
        # detect possible changes in all related movements
        min_angle_change_detection[angle_name_] = min(changes_in_movements)
    return min_angle_change_detection


def cluster_angles_over_train_files(data_local_dir, rdp_epsilon):
    amass_clip_lst = []
    amass_mocap_path_lst = []
    for cur_path, directories, file_names in os.walk(os.path.join(data_local_dir, 'AMASS_Data')):
        for file_name in file_names:
            if file_name.split('.')[-1] == 'npz':
                amass_mocap_path_lst.append(os.path.join(cur_path, file_name))

    for file_path in amass_mocap_path_lst:
        amass_clip = AmassClip(data_path=file_path, angle_order_lst=deepcopy(joint_angle_order_lst))
        amass_clip.segment_joint_angle_frms()   # todo, better epsilon
        # amass_file.visualize_segmentation()
        amass_clip_lst.append(amass_clip)
        for angle_ix, angle in enumerate(amass_clip.angle_order_lst):
            angle_cluster_order_lst[angle_ix].add_joint_segments(angle.segment_value_in_clip)

    # after collecting segments of all training files, cluster the segments
    for joint_cluster in angle_cluster_order_lst:
        joint_cluster.cluster_training()
    # represent single file as discrete cluster of angles, generate training data for RL
    for amass_clip in amass_clip_lst:
        for angle, angle_cluster in zip(amass_clip.angle_order_lst, angle_cluster_order_lst):
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
    amass_joint_order_lst = [
        'left_hip', 'right_hip', 'waist', 'left_knee', 'right_knee', 'spine', 'left_ankle', 'right_ankle',
        'spine1', 'left_toe', 'right_toe', 'spine2', 'left_clavicle', 'right_clavicle', 'neck', 'head',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist'
    ]
    angle_type_lst = ['rotation', 'extension', 'abduction']
    angle_cluster_order_lst = []
    joint_angle_order_lst = []
    for joint in amass_joint_order_lst:
        for joint_comp in angle_type_lst:
            angle_name = '_'.join([joint, joint_comp])
            angle_cluster_order_lst.append(AngleCluster(angle_name))
            joint_angle_order_lst.append(JointAngle(angle_name))

    annotation_train_path = os.path.join(os.getcwd(), 'babel_v1.0_release', 'train.json')
    movement_segment_dict = extract_movement_segments(
        data_local_dir=local_dir,
        annotation_file_path=annotation_train_path
    )
    angle_clip_base = detect_angle_changes_over_train(movement_segment_dict)
    cluster_angles_over_train_files(local_dir, angle_clip_base)





