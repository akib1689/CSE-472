from custom_logistic_regression import CustomLogisticRegression
import numpy as np

class AdaBoost:
    def __init__(self, num_hypothesis=10, verbose=False, early_stopping_threshold=0.5):
        # list of hypothesis
        self.hypothesis = []
        # list of hypothesis weights
        self.hypothesis_weights = []
        self.num_hypothesis = num_hypothesis
        self.verbose = verbose
        self.early_stopping_threshold = early_stopping_threshold
        
        
    def resample(self, x, y, weights):
        # resample the dataset
        indices = np.random.choice(len(x), len(x), p=weights, replace=True)
        return x[indices], y[indices]
        
    def fit(self, x, y, num_iterations=15):
        # initialize the weights of each input sample in x
        weights = np.ones(len(x)) / len(x)
        
        for i in range(num_iterations):
            if self.verbose:
                print('Iteration:', i + 1)
            # resample the dataset
            x_sample, y_sample = self.resample(x, y, weights)
            
            # create a new hypothesis
            hypothesis = CustomLogisticRegression(early_stopping_threshold=self.early_stopping_threshold, verbose=self.verbose)
            
            # fit the hypothesis
            hypothesis.fit(x_sample, y_sample)
            
            # predict the labels
            y_pred = hypothesis.predict(x)
            
            # calculate the error
            error = np.sum(weights * (y_pred != y))
            
            if error > 0.5:
                continue
            
            # calculate the hypothesis weight
            for j in range(len(weights)):
                if y_pred[j] == y_sample[j]:
                    weights[j] *= error / (1 - error)
                    
            # normalize the weights
            weights /= np.sum(weights)
            
            # store the hypothesis
            self.hypothesis.append(hypothesis)
            
            # store the hypothesis weight
            self.hypothesis_weights.append(np.log((1 - error) / error))
            
            if self.verbose:
                print('Number of hypothesis:', len(self.hypothesis))
            
            # early stopping
            if len(self.hypothesis) == self.num_hypothesis:
                break
            
    def predict(self, x):
        # return the weighted sum of the predictions of the hypothesis
        return np.sign(np.sum([self.hypothesis_weights[i] * self.hypothesis[i].predict(x) for i in range(len(self.hypothesis))], axis=0))