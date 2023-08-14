"""
utilities
"""
import numpy as np
from enum import IntEnum
from numpy.linalg import norm
from transforms3d import quaternions as quat
from sklearn.cluster import AgglomerativeClustering, KMeans


class AMASSJoint(IntEnum):
    # ROOT = 0
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
    SPINE2_JOINT = 22  # dummy joint to match with current skeleton


JointType = IntEnum('JointType', ['ROTATION', 'EXTENSION', 'ABDUCTION'])


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


ANGLE_CLUSTER_ORDER_LST = []
JOINT_ANGLE_ORDER_LST = []
for joint in AMASSJoint:
    for joint_type in JointType:
        angle_name = '_'.join([joint.name, joint_type.name])
        ANGLE_CLUSTER_ORDER_LST.append(AngleCluster(angle_name))
        JOINT_ANGLE_ORDER_LST.append(JointAngle(angle_name))
