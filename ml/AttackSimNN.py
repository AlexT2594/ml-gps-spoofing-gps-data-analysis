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

from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Dropout


def main():

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

    #numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer()),
    #                                     ('scaler', StandardScaler())])

    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer())])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)])

    data_file_name = "../data/attack_sim_sv_info_DOP_analysis.csv"
    data = pd.read_csv(data_file_name)

    X = data.drop('spoofed', axis=1)
    y = data['spoofed']

    X = preprocessor.fit_transform(X)

    y = y.values

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    n_cols = X_train.shape[1]

    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=n_cols))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=10, callbacks=[EarlyStopping(monitor='loss', patience=3)])

    score = model.evaluate(X_test, y_test, batch_size=10)

    print("\nScore: %.3f" % score[1])


if __name__ == '__main__':
    main()
