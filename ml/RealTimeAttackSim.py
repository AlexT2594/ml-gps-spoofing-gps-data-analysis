import pandas as pd
from pandas.compat import StringIO

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from kafka import KafkaConsumer

from utils.data import gen_test_entry

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import datetime as dt
from dateutil import tz

from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Dropout

import multiprocessing as mp
from threading import Thread

ALG_LEN = 3

names = ["Logistic Regression", "Linear SVM", "Neural Net"]

topic_name = 'raw_nmea_numeric'

from_zone = tz.tzutc()
to_zone = tz.tzlocal()


def main(file):

    # Create figure for plotting
    fig = plt.figure(figsize=(7, 3))
    axes = []
    for i in range(1, ALG_LEN + 1):
        axes.append(fig.add_subplot(1, 3, i))

    plt.subplots_adjust(top=0.8, bottom=0.25, left=0.17, hspace=1.1, wspace=0.3)

    N_SAT = 32

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

    X_train_NN = preprocessor.fit_transform(X_train)
    y_train_NN = y_train.values

    n_cols = X_train_NN.shape[1]

    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=n_cols))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    classifiers_init = [
        LogisticRegression(solver="liblinear", multi_class='auto'),
        SVC(kernel="linear", C=0.025),
        model
    ]

    classifiers = []



    # iterate over classifiers
    for name, classifier in zip(names, classifiers_init):
        #print("==> Classifier: " + name)
        #print("\tTraining phase\n")

        if name == "Neural Net":
            classifier.fit(X_train_NN, y_train_NN, epochs=10, callbacks=[EarlyStopping(monitor='loss', patience=3)])
            classifier._make_predict_function()
            classifiers.append(classifier)
        else:
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

    consumer = KafkaConsumer(topic_name, auto_offset_reset='earliest', bootstrap_servers=['localhost:9093'],
                             security_protocol='SSL', ssl_check_hostname=False,
                             ssl_cafile='kafka_ssl/CARoot.pem',
                             ssl_certfile='kafka_ssl/certificate.pem',
                             ssl_keyfile='kafka_ssl/key.pem',
                             consumer_timeout_ms=10000)

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

            for name, clf in zip(names, classifiers):
                if name == "Neural Net":
                    pred = clf.predict(X_test)[0]
                else:
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
    main("../data/numeric_eval/data_train.csv")
