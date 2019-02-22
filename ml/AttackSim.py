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

def main():

    N_SAT = 32

    names = ["Logistic Regression", "Nearest Neighbors", "Linear SVM", "RBF SVM",
             "Decision Tree", "Random Forest", "Neural Net", "AdaBoost"]

    classifiers = [
        LogisticRegression(solver="liblinear", multi_class='auto'),
        KNeighborsClassifier(),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(solver="lbfgs"),
        AdaBoostClassifier()
    ]

    # We create the preprocessing pipelines for both numeric and categorical data.
    numeric_features = ['time_sin', 'time_cos', 'lat_sin', 'lat_cos', 'long_sin', 'long_cos', 'n_satellites']
    # numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
    # ('scaler', StandardScaler())])
    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer())])

    categorical_features = []
    for i in range(1, N_SAT + 1):
        category = "sv_prn_" + str(i)
        categorical_features.append(category)
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer()),
                                              ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    data_train_file_name = "../data/spoofed_data_shuffled2.csv"
    data_train = pd.read_csv(data_train_file_name)

    X_train = data_train.drop('spoofed',axis=1)
    y_train = data_train['spoofed']

    data_test_file_name = "../data/day40attack.csv"
    data_test = pd.read_csv(data_test_file_name)

    X_test = data_test.drop('spoofed', axis=1)
    y_test = data_test['spoofed']

    # iterate over classifiers
    for name, classifier in zip(names, classifiers):

        if name == "Decision Tree":
            print("==> Classifier: " + name)
            clf = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('classifier', classifier)])

            clf.fit(X_train, y_train)
            preds= clf.predict(X_test)

            scores = {0: 0, 1: 0, 2: 0}
            for prediction in preds:
                scores[prediction] += 1

            print(scores)

            score = accuracy_score(y_test, preds)
            print("Accuracy:   %0.3f" % score)




if __name__ == '__main__':
    main()
