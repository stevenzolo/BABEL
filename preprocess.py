"""
a.k.a. dimension reduction in current version
correlation analysis between angle variance (or other metric) and action label, select the important angles
"""
import os
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from matplotlib import pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.neural_network import MLPClassifier

from utils import JOINT_ANGLE_ORDER_LST
from data_preparation import extract_movement_segments

import warnings
warnings.filterwarnings("ignore")


def select_related_angles(movement_segment_dict_, regulation_fac=0.1, plot=False):
    movement_segments_df = {}
    for joint in JOINT_ANGLE_ORDER_LST:
        movement_segments_df[joint.name] = []
    movement_segments_df['movement'] = []
    for movement_lbl in movement_segment_dict_:
        for movement in movement_segment_dict_[movement_lbl]:
            for ix, joint in enumerate(JOINT_ANGLE_ORDER_LST):
                # variance as array fluctuation measures, maybe not suitable
                movement_segments_df[joint.name].append(np.var(movement.pose[:, ix]))
            movement_segments_df['movement'].append(movement_lbl)
    movement_segments_df = pd.DataFrame(movement_segments_df)

    train_accuracy_lst = []
    selected_features_lst = []
    classifier_lst = [
        SVC(random_state=101),
        MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    ]
    x_feature = movement_segments_df.values[:, :-1]
    y_label = movement_segments_df.values[:, -1]  # movements
    for select_angle_num in range(1, len(JOINT_ANGLE_ORDER_LST) + 1):
        select_best = SelectKBest(f_classif, k=select_angle_num)
        x_new = select_best.fit_transform(x_feature, y_label)
        selected_features = select_best.get_feature_names_out(movement_segments_df.columns.to_list()[:-1])
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
    local_dir = os.path.join(os.getcwd(), 'AMASS_Data')
    annotation_train_path = os.path.join(os.getcwd(), 'babel_v1.0_release', 'train.json')
    movement_segment_dict = extract_movement_segments(
        data_local_dir=local_dir,
        annotation_file_path=annotation_train_path
    )   # one segment sometimes attached multiple labels
    optimal_feature_set = select_related_angles(movement_segment_dict)




