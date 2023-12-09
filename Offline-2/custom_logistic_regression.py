import numpy as np

class CustomLogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=10000, early_stopping_threshold=0.5, num_features=None, verbose=False, fit_intercept=True):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.early_stopping_threshold = early_stopping_threshold
        self.num_features = num_features
        self.verbose = verbose
        self.fit_intercept = fit_intercept
        self.top_features_indices = None
        
        
    def entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def information_gain(self, X, y, feature_index):
        parent_entropy = self.entropy(y)
        median = np.median(X[:, feature_index])
        left_indices = np.where(X[:, feature_index] <= median)
        right_indices = np.where(X[:, feature_index] > median)
        left_entropy = self.entropy(y[left_indices])
        right_entropy = self.entropy(y[right_indices])
        num_left = len(left_indices[0]) / len(y)
        num_right = len(right_indices[0]) / len(y)
        child_entropy = num_left * left_entropy + num_right * right_entropy
        return parent_entropy - child_entropy

    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def __sigmoid(self, z):
        # Limit the values in z to avoid overflow
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))

    def __loss(self, h, y):
        # Add small constants to avoid taking the log of zero
        return (-y * np.log(h + 1e-10) - (1 - y) * np.log(1 - h + 1e-10)).mean()

    def fit(self, X, y):
        if self.fit_intercept:
            X = self.__add_intercept(X)

         # Select the top features based on information gain
        if self.num_features is not None and self.num_features < X.shape[1] - 1 and self.num_features > 0:
            information_gain_values = np.array([self.information_gain(X, y, i) for i in range(X.shape[1])])
            self.top_features_indices = np.argsort(-information_gain_values)[:self.num_features]
            X = X[:, self.top_features_indices]

        # weights initialization
        self.theta = np.zeros(X.shape[1])

        for i in range(self.num_iterations):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.learning_rate * gradient

            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            loss = self.__loss(h, y)

            if self.verbose and i % 10000 == 0:
                print(f'loss: {loss} \t')

            if loss < self.early_stopping_threshold:
                if self.verbose:
                    print(f'early stopping with loss: {loss} \t')
                break

    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)

        if self.num_features is not None and self.num_features < X.shape[1] - 1 and self.num_features > 0:
            X = X[:, self.top_features_indices]

        return self.__sigmoid(np.dot(X, self.theta))

    def predict(self, X, threshold=0.5):
        return self.predict_prob(X) >= threshold