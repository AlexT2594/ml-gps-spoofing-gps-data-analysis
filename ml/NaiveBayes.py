from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
from utils.ml import BasicCountVectorizer
from utils.data import get_data_from_file_2, get_single_elem_from_file, get_data_from_file, transform_data_into_CSV


def main():
    print("==> Naive Bayes")
    dataset, dataset_labels = get_data_from_file_2("../data/work_route_sample_stable.txt")

    print("==> Printing dataset")
    print(dataset[0])
    print(str(len(dataset[0])))

    transform_data_into_CSV("../data/work_route_sample_stable.txt")

'''
    print("==> Printing dataset labels")
    print(dataset_labels)

    data_train, data_test, y_train, y_test = train_test_split(dataset, dataset_labels)

    print("==> Printing X_train")
    print(data_train)

    print("==> Printing X_test")
    print(data_test)

    vectorizer = BasicCountVectorizer()
    vectorizer.fit(data_train)
    X_train = vectorizer.transform(data_train)
    X_test = vectorizer.transform(data_test)

    print("==> Printing tokens")
    print(vectorizer.get_param_names())

    print("==> Number of unique tokens: " + str(len(vectorizer.get_param_names())))

    print("==> Printing X_train")
    print(X_train)

    clf = MultinomialNB()
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)

    score = metrics.accuracy_score(y_test, pred)
    print("Accuracy:   %0.3f" % score)

    print("==> Confusion matrix")
    print(metrics.confusion_matrix(y_test, pred))

    return
'''

if __name__ == '__main__':
    main()
