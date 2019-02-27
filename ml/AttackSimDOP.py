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

    # We create the preprocessing pipelines for numeric data
    numeric_features = ["PDOP", "HDOP", "VDOP"]
    # numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
    # ('scaler', StandardScaler())])

    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer())])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)])

    data_file_name = "../data/attack_sim_DOP_analysis.csv"
    data = pd.read_csv(data_file_name)

    X = data.drop('spoofed', axis=1)
    y = data['spoofed']

    # iterate over classifiers
    for name, classifier in zip(names, classifiers):
        clf = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', classifier)])

        """

        This was done traditionally, without using a K-fold cross-validation method.

        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        print("==> Classifier: " + name)
        print("\tScore: %.3f" % score)
        """

        #  mean score and the 95% confidence interval of the score
        scores = cross_val_score(clf, X, y, cv=StratifiedKFold(n_splits=10, shuffle=True))
        print("==> Classifier: " + name)
        print("\tAccuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))


if __name__ == '__main__':
    main()