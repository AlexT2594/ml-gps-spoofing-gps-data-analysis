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
from dateutil import tz
import time
import random

import multiprocessing as mp
from threading import Thread

ALG_LEN = 8

names = ["Logistic Regression", "Nearest Neighbors", "Linear SVM", "RBF SVM",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost"]

topic_name = 'raw_nmea'

from_zone = tz.tzutc()
to_zone = tz.tzlocal()


def main(file):

    # Create figure for plotting
    fig = plt.figure(figsize=(10, 5))
    axes = []
    for i in range(1, ALG_LEN + 1):
        axes.append(fig.add_subplot(2, 4, i))

    plt.subplots_adjust(left=0.15, bottom=0.2, hspace=1.1, wspace=0.3)

    N_SAT = 32

    classifiers_init = [
        LogisticRegression(solver="liblinear", multi_class='auto'),
        KNeighborsClassifier(),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(solver="lbfgs"),
        AdaBoostClassifier()
    ]

    classifiers = []

    # We create the preprocessing pipelines for numeric data
    numeric_features = []
    # numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
    # ('scaler', StandardScaler())])

    for i in range(1, N_SAT + 1):
        numeric_features.append("sv_elev_" + str(i))
        numeric_features.append("sv_azimuth_" + str(i))
        numeric_features.append("sv_snr_" + str(i))

    numeric_features += ["PDOP", "HDOP", "VDOP"]

    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer())])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)])

    data_file_name = file
    #data_file_name = "../data/day39_40_spoof_with_true_data1.csv"
    data = pd.read_csv(data_file_name)

    X_train = data.drop('spoofed', axis=1)
    y_train = data['spoofed']

    # iterate over classifiers
    for name, classifier in zip(names, classifiers_init):
        #print("==> Classifier: " + name)
        #print("\tTraining phase\n")
        clf = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', classifier)])

        clf.fit(X_train, y_train)

        classifiers.append(clf)

    xs = []
    ys = [[], [], [], [], [], [], [], []]
    queue = mp.Queue()

    thread = Thread(target=consumeData, args=(queue, classifiers))
    thread.start()

    # Set up plot to call animate() function periodically
    ani = animation.FuncAnimation(fig, animate, fargs=(fig, axes, queue, xs, ys), interval=1000)
    plt.show()

    print("    Exiting...")
    return


def animate(i, fig, axes, q, xs, ys):
    #xs.append(dt.datetime.now().strftime('%H:%M:%S'))

    try:
        time = q.get(timeout=10)
        predictions = q.get()
    except:
        print("    Timeout expired, closing visualization.")
        plt.close(fig)
        return

    if time is not "":
        time_beautify = time[:2] + ":" + time[2:4] + ":" + time[4:6]
        utc = dt.datetime.strptime(time_beautify, "%H:%M:%S")
        utc = utc.replace(tzinfo=from_zone)

        time = utc.astimezone(to_zone).strftime("%H:%M:%S")


    xs.append(time)
    xs = xs[-10:]
    min_time = ''
    for xs_elem in xs:
        if xs_elem is not '':
            min_time = xs_elem
            break
    max_time = xs[-1]

    x_labels = [''] * len(xs)

    for xs_index in range(len(xs)):
        if xs[xs_index] is min_time or xs[xs_index] is max_time:
            x_labels[xs_index] = xs[xs_index]

    for i in range(ALG_LEN):
        ys[i].append(predictions[i])

        # Limit x and y lists to 20 items
        ys[i] = ys[i][-10:]

        axes[i].clear()
        axes[i].plot(xs, ys[i])

        axes[i].set_xticklabels(x_labels)

        axes[i].set_yticks([-3, -2, -1, 0, 1, 2, 3])
        axes[i].set_title(names[i].replace(' ', '\n'))

        axes[i].tick_params(axis='x', rotation=45)

        if i != 0 and i != 4:
            axes[i].set_yticklabels(['', '', '', '', '', '', ''])
        else:
            axes[i].set_yticklabels(['', 'Not Stable', '', 'Safe', '', 'Spoof Attack!', ''])


def consumeData(queue, classifiers):

    consumer = KafkaConsumer(topic_name, auto_offset_reset='earliest', bootstrap_servers=['localhost:9092']
                             , consumer_timeout_ms=1000)

    for msg in consumer:
        entry = msg.value.decode('utf-8')

        data_test, time = gen_test_entry(entry)

        #we'll update the queue in the following way:
        #the first added element is the x_axis elem
        #the second element added is the array containing the y_axis elems

        if data_test == "":
            predictions = [-2] * 8
        elif data_test == "incomplete":
            continue
        else:
            X_test = pd.read_csv(StringIO(data_test))

            predictions = []

            for clf in classifiers:
                pred = clf.predict(X_test)
                if pred[0] == 0:
                    pred = 0
                else:
                    pred = 2

                predictions.append(pred)

        queue.put(time)
        queue.put(predictions)

    consumer.close()


if __name__ == '__main__':
    main("../data/day39_40_spoof_with_true_data1.csv")
