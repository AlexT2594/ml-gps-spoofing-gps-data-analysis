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
from sklearn.model_selection import cross_val_score

def main():

    N_SAT = 32
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
    # - sv_prn_i: categories encoded as integers {0, 1}.

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

    X = data.drop('spoofed', axis=1)
    y = data['spoofed']

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    print(X_train)

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
        scores = cross_val_score(clf, X, y, cv=10)
        print("==> Classifier: " + name)
        print("\tAccuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))


if __name__ == '__main__':
    main()
