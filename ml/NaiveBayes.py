from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
from utils.ml import BasicCountVectorizer
from utils.data import get_data_from_file_2, get_single_elem_from_file, get_data_from_file, transform_data_into_CSV, transform_data_for_numeric_into_CSV
from sklearn.model_selection import cross_val_score


def main():
    dataset, y = get_data_from_file_2("../data/work_route_sample_stable.txt")

    vectorizer = BasicCountVectorizer()
    vectorizer.fit(dataset)
    X = vectorizer.transform(dataset)

    clf = MultinomialNB()

    #  mean score and the 95% confidence interval of the score
    scores = cross_val_score(clf, X, y, cv=10)
    print("==> Classifier: Naive Bayes")
    print("\tAccuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))

"""
    This was done before not using K-fold cross-validation
    
    data_train, data_test, y_train, y_test = train_test_split(dataset, dataset_labels)
    
    vectorizer = BasicCountVectorizer()
    vectorizer.fit(data_train)
    X_train = vectorizer.transform(data_train)
    X_test = vectorizer.transform(data_test)

    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)
    score = metrics.accuracy_score(y_test, pred)
    print("Accuracy:   %0.3f" % score)

    print("==> Confusion matrix")
    print(metrics.confusion_matrix(y_test, pred))
"""


if __name__ == '__main__':
    main()
