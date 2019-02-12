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

def main():

    N_SAT = 16
    # Read data from Titanic dataset.
    data_file_name = "../data/work_route_sample_stable_numeric.csv"
    data = pd.read_csv(data_file_name)

    names = ["Logistic Regression", "Nearest Neighbors", "Linear SVM", "RBF SVM",
             "Decision Tree", "Random Forest", "Neural Net", "AdaBoost"]

    classifiers = [
        LogisticRegression(solver="liblinear"),
        KNeighborsClassifier(),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(solver="lbfgs"),
        AdaBoostClassifier()
    ]

    # We will train our classifier with the following features:
    # Numeric Features:
    # - time_sin: float.
    # - time_cos: float.
    # - lat_sin: float.
    # - lat_cos: float.
    # - long_sin: float.
    # - long_cos: float.
    # - n_satellites: int
    # Categorical Features:
    # - sat_prn_i: categories encoded as integers{01, 02, ..., 15}.

    # We create the preprocessing pipelines for both numeric and categorical data.
    numeric_features = ['time_sin', 'time_cos', 'lat_sin', 'lat_cos', 'long_sin', 'long_cos', 'n_satellites']
    # numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
    # ('scaler', StandardScaler())])
    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer())])

    categorical_features = []
    for i in range(N_SAT):
        category = "sat_prn_" + str(i)
        categorical_features.append(category)
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer()),
                                              ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    X = data.drop('spoofed', axis=1)
    y = data['spoofed']

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    print(X_train)

    # iterate over classifiers
    for name, classifier in zip(names, classifiers):
        clf = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', classifier)])

        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        print("==> Classifier: " + name)
        print("\tScore: %.3f" % score)


if __name__ == '__main__':
    main()
