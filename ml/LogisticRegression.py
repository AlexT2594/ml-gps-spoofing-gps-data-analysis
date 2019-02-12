import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def main():

    N_SAT = 16
    # Read data from Titanic dataset.
    data_file_name = "../data/work_route_sample_stable_numeric.csv"
    data = pd.read_csv(data_file_name)

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
    #numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
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

    # Append classifier to preprocessing pipeline.
    # Now we have a full prediction pipeline.
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', LogisticRegression(solver='liblinear'))])

    X = data.drop('spoofed', axis=1)
    y = data['spoofed']

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    print(X_train)

    clf.fit(X_train, y_train)
    print("model score: %.3f" % clf.score(X_test, y_test))


if __name__ == '__main__':
    main()
