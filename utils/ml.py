class BasicCountVectorizer:
    vocabulary = None

    def __init__(self):
        self.vocabulary = set()

    def fit(self, X):
        """
        Fit Basic Count Vectorizer according to X
        :param X: {array-like, matrix}, shape = [n_samples, n_features]
             Training vectors, where n_samples is the number of samples and
             n_features is the number of features.
        :return: self : object
        """

        for sample in X:

            for token in sample:
                self.vocabulary.add(token)

    def transform(self, X):
        """
        Counts the tokens present in the parameter according to the vocabulary
        :param X: {array-like, sparse matrix}, shape = [n_samples, n_features]
             Training vectors, where n_samples is the number of samples and
             n_features is the number of features.
        :return: {array-like, dense matrix}, shape = [n_samples, n_features]
            Each element represents the number of times the token is present in
            the sample.
        """

        token_matrix = []

        for sample in X:
            sample_dict = {}

            for token in sample:
                if sample_dict.get(token) is None:
                    sample_dict[token] = 1
                else:
                    sample_dict[token] = sample_dict[token] + 1

            sample_array = []
            for word in self.vocabulary:
                if sample_dict.get(word) is None:
                    sample_array.append(0)
                else:
                    sample_array.append(sample_dict.get(word))

            token_matrix.append(sample_array)

        return token_matrix

    def get_param_names(self):
        return self.vocabulary

