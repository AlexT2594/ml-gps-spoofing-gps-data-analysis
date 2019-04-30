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
import random

from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Dropout

def main():

    N_SAT = 32

    names = ["Logistic Regression", "Linear SVM", "Neural Net"]

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

    data_train_file_name = "../data/numeric_eval/data_train.csv"
    data_train = pd.read_csv(data_train_file_name)

    X_train = data_train.drop('spoofed', axis=1)
    #y_train = y_train_NN = [random.randint(0, 1) for _ in range(len(data_train['spoofed']))]
    y_train = data_train['spoofed']

    data_test_file_name = "../data/numeric_eval/data_test.csv"
    data_test = pd.read_csv(data_test_file_name)

    X_test = data_test.drop('spoofed', axis=1)
    y_test = data_test['spoofed']

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


    # iterate over classifiers
    for name, classifier in zip(names, classifiers_init):
        print("==> Classifier: " + name)

        if name == "Neural Net":
            classifier.fit(X_train_NN, y_train_NN, epochs=10, callbacks=[EarlyStopping(monitor='loss', patience=3)])
            classifier._make_predict_function()
            clf = classifier

        else:
            clf = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('classifier', classifier)])
            clf.fit(X_train, y_train)



        pred = clf.predict(X_test)

        #if name == "Neural Net":
        #    pred = [round(_[0]) for _ in clf.predict(X_test)]


        score = accuracy_score(y_test, pred)
        print("\tAccuracy:   %0.3f\n" % score)


if __name__ == '__main__':
    main()
