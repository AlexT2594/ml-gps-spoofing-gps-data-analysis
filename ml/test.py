import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix

from utils.data import get_numeric_data_from_file
from random import shuffle

from pandas.compat import StringIO
from kafka import KafkaConsumer
from utils.data import gen_test_entry

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import datetime as dt
import time
import random

from utils.data import concatenate_files, transform_data_for_sv_info_DOP_analysis_into_CSV

import multiprocessing as mp


def main():
    transform_data_for_sv_info_DOP_analysis_into_CSV(["../data/ground_truth.txt",
                                                      "../data/lat_long_eval/spoof_70_A.txt",
                                                      "../data/lat_long_eval/spoof_70_B.txt",
                                                      "../data/lat_long_eval/spoof_70_C.txt",
                                                      "../data/lat_long_eval/spoof_70_E.txt",
                                                      "../data/lat_long_eval/spoof_70_F.txt",
                                                      "../data/lat_long_eval/spoof_70_G.txt",
                                                      "../data/lat_long_eval/spoof_70_H.txt"],
                                                     [0, 1, 1, 1, 1, 1, 1, 1],
                                                     False, "../data/numeric_eval/data_train.csv")

    transform_data_for_sv_info_DOP_analysis_into_CSV(["../data/numeric_eval/true_11am_10min.txt",
                                                      "../data/numeric_eval/spoof_71_A.txt"],
                                                     [0, 1],
                                                     False, "../data/numeric_eval/data_test.csv")


if __name__ == '__main__':
    main()
