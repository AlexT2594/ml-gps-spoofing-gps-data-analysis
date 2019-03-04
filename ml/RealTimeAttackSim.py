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

import multiprocessing as mp

names = ["Logistic Regression", "Nearest Neighbors", "Linear SVM", "RBF SVM",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost"]

topic_name = 'raw_nmea'

# Create figure for plotting
fig, (ax1, ax2) = plt.subplots(2, 1)
#plt.xticks(rotation=45, ha='right')
ax1.tick_params(axis='x', rotation=45)
ax2.tick_params(axis='x', rotation=45)
plt.subplots_adjust(left=0.2, bottom=0.2, hspace=0.9)



def main():
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

    data_file_name = "../data/day39_40_spoof_with_true_data1.csv"
    data = pd.read_csv(data_file_name)

    X_train = data.drop('spoofed', axis=1)
    y_train = data['spoofed']

    # iterate over classifiers
    for name, classifier in zip(names, classifiers_init):
        print("==> Classifier: " + name)
        print("\tTraining phase\n")
        clf = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', classifier)])

        clf.fit(X_train, y_train)

        classifiers.append(clf)

    queue = mp.Queue()
    consumer_process = mp.Process(target=consumeData, args=(queue, classifiers))
    consumer_process.start()

    xs = []
    ys = [[], []]

    # Set up plot to call animate() function periodically
    ani = animation.FuncAnimation(fig, animate, fargs=(queue, xs, ys), interval=1000)
    plt.show()


def animate(i, q, xs, ys):
    xs.append(dt.datetime.now().strftime('%H:%M:%S'))
    predictions = q.get()
    ys[0].append(predictions[0])
    ys[1].append(predictions[1])

    # Limit x and y lists to 20 items
    xs = xs[-20:]
    ys[0] = ys[0][-20:]
    ys[1] = ys[1][-20:]

    # Draw x and y lists
    ax1.clear()
    ax1.plot(xs, ys[0])

    ax2.clear()
    ax2.plot(xs, ys[1])

    # Format plot
    ax1.set_title('Logistic Regression Analysis')
    ax1.set_yticks([-2, -1, 0, 1, 2])
    ax1.set_yticklabels(['', 'Safe', '', 'Spoof Attack!', ''])

    ax2.set_title("Nearest Neighbors")
    ax2.set_yticks([-2, -1, 0, 1, 2])
    ax2.set_yticklabels(['', 'Safe', '', 'Spoof Attack!', ''])


def consumeData(queue, classifiers):
    consumer = KafkaConsumer(topic_name, auto_offset_reset='earliest', bootstrap_servers=['localhost:9092'])
                             #,consumer_timeout_ms=1000)

    for msg in consumer:
        entry = msg.value.decode('utf-8')

        data_test = gen_test_entry(entry)
        if data_test == "":
            continue

        X_test = pd.read_csv(StringIO(data_test))

        clf = classifiers[0]
        pred1 = clf.predict(X_test)
        if pred1[0] == 0:
            pred1 = -1
        else:
            pred1 = 1

        clf = classifiers[1]
        pred2 = clf.predict(X_test)
        if pred2[0] == 0:
            pred2 = -1
        else:
            pred2 = 1

        queue.put([pred1, pred2])


if __name__ == '__main__':
    main()
