"""
a.k.a. dimension reduction in current version
correlation analysis between angle variance (or other metric) and action label, select the important angles
"""
import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from matplotlib import pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.neural_network import MLPClassifier

from utils import JOINT_ANGLE_ORDER_LST
from data_preparation import extract_movement_segments, MovementSegment

import warnings
warnings.filterwarnings("ignore")


def extract_feature_from_move(move_seg_dict_path_, local_dir_, annotation_path_, move_seg_df_path_):
    # variance as feature in current version
    if not os.path.exists(move_seg_dict_path_):
        move_seg_dict_ = extract_movement_segments(
            data_local_dir=local_dir_,
            annotation_file_path=annotation_path_,
            save_pkl_path=move_seg_dict_path_
        )   # one segment sometimes attached multiple labels
    else:
        with open(move_seg_dict_path_, 'rb') as msp:
            move_seg_dict_ = pickle.load(msp)

    movement_segments_df_ = {}
    for joint in JOINT_ANGLE_ORDER_LST:
        movement_segments_df_[joint.name] = []
    movement_segments_df_['movement'] = []
    for movement_lbl in move_seg_dict_:
        for movement in move_seg_dict_[movement_lbl]:
            for ix, joint in enumerate(JOINT_ANGLE_ORDER_LST):
                # variance as array fluctuation measures, maybe not suitable
                movement_segments_df_[joint.name].append(np.var(movement.pose[:, ix]))
            movement_segments_df_['movement'].append(movement_lbl)
    movement_segments_df_ = pd.DataFrame(movement_segments_df_)
    movement_segments_df_.to_csv(move_seg_df_path_)
    return movement_segments_df_


def select_related_angles(train_move_segment_df_, val_move_segment_df_, regulation_fac=0.1, plot=False):
    train_accuracy_lst = []
    selected_features_lst = []
    classifier_lst = [
        SVC(random_state=101),
        MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    ]
    train_feature = train_move_segment_df_.values[:, :-1]
    train_label = train_move_segment_df_.values[:, -1]  # movements
    for select_angle_num in range(1, len(JOINT_ANGLE_ORDER_LST) + 1):
        select_best = SelectKBest(f_classif, k=select_angle_num)
        x_new = select_best.fit_transform(train_feature, train_label)
        selected_features = select_best.get_feature_names_out(train_move_segment_df_.columns.to_list()[:-1])
        selected_features_lst.append(selected_features)
        for clf in classifier_lst:
            clf.fit(x_new, y_label)
            train_accuracy_lst.append(
                clf.score(x_new, y_label) - regulation_fac * select_angle_num / len(JOINT_ANGLE_ORDER_LST)
            )

    if plot:
        plt.figure()
        for rm in range(len(classifier_lst)):
            plt.plot(
                [train_accuracy_lst[ix] for ix in range(len(train_accuracy_lst)) if ix % len(classifier_lst)==rm],
                label=repr(type(classifier_lst[rm])).split('.')[-1][:-2]
            )
        plt.xlabel('selected angle numbers')
        plt.ylabel('train accuracy')
        plt.legend()
        plt.show()
    best_train_acc = max(train_accuracy_lst)
    best_features = selected_features_lst[int(np.argmax(train_accuracy_lst) / len(classifier_lst))]
    best_clf = classifier_lst[np.argmax(train_accuracy_lst) % len(classifier_lst)]
    print("Best train accuracy: {}; \ncorresponding features: {}; \nclassifier: {}".format(
        round(best_train_acc, 3), best_features, best_clf
    ))  # best accuracy ~0.35 from SVM (linear), todo, need more accurate ?
    return best_features


if __name__ == "__main__":
    # local_dir = os.path.join(os.getcwd(), 'AMASS_Data')
    local_dir = os.path.join('/media/yan/TOSHIBA EXT', 'AMASS_Data')  # 'lsblk' cmd list all disk names
    train_movement_segment_df = extract_feature_from_move(
        move_seg_dict_path_=os.path.join('/media/yan/TOSHIBA EXT', 'BabelData', 'train_move_seg.pkl'),
        # move_seg_dict_path_='train_move_seg.pkl',
        local_dir_=local_dir,
        annotation_path_=os.path.join(os.getcwd(), 'babel_v1.0_release', 'train.json'),
        move_seg_df_path_=os.path.join('/media/yan/TOSHIBA EXT', 'BabelData', 'train_move_df.csv')
        # move_seg_df_path_='train_move_df.csv'
    )
    val_movement_segment_df = extract_feature_from_move(
        move_seg_dict_path_=os.path.join('/media/yan/TOSHIBA EXT', 'BabelData', 'val_move_seg.pkl'),
        local_dir_=local_dir,
        annotation_path_=os.path.join(os.getcwd(), 'babel_v1.0_release', 'val.json'),
        move_seg_df_path_=os.path.join('/media/yan/TOSHIBA EXT', 'BabelData', 'val_move_df.csv')
    )
    # optimal_feature_set = select_related_angles(
    #     train_movement_segment_df, val_movement_segment_df)




