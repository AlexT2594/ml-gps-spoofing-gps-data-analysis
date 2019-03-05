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
fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
#plt.xticks(rotation=45, ha='right')
ax1.tick_params(axis='x', rotation=45)
ax2.tick_params(axis='x', rotation=45)
plt.subplots_adjust(left=0.2, bottom=0.2, hspace=2.0)



def main():


    queue = mp.Queue()

    consumer_process = mp.Process(target=consumeData, args=(queue,))
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

    ax3.clear()
    ax3.plot(xs, ys[1])

    # Format plot
    ax1.set_title('Logistic Regression Analysis')
    ax1.set_yticks([-2, -1, 0, 1, 2])
    ax1.set_yticklabels(['', 'Safe', '', 'Spoof Attack!', ''])

    ax2.set_title("Nearest Neighbors")
    ax2.set_yticks([-2, -1, 0, 1, 2])
    ax2.set_yticklabels(['', 'Safe', '', 'Spoof Attack!', ''])

    ax3.set_title("Nearest Neighbors")
    ax3.set_yticks([-2, -1, 0, 1, 2])
    ax3.set_yticklabels(['', 'Safe', '', 'Spoof Attack!', ''])


def consumeData(queue):
    while True:
        queue.put([1, -1])
        queue.put([-1, 1])


if __name__ == '__main__':
    main()
