from sklearn.base import clone, BaseEstimator
from sklearn.tree import DecisionTreeRegressor
from sklearn.dummy import DummyRegressor

import numpy as np
from scipy.special import expit


def mse_gradient(y, f):
    """ Least Squares gradient """
    return y - f


def lad_gradient(y, f):
    """ Also known as MAE. 1.0 if y - pred > 0.0 else -1.0 """
    return 2.0 * (y - f > 0) - 1.0


def logistic_gradient(y, f):
    """ expit(x) = 1/(1+exp(-x)). It is the inverse of the logit function. """
    return y - expit(f)


class GBoost(BaseEstimator):
    """
    Basic implementation of Gradient Boosting that performs regression with 
    MSE or MAE loss functions, and classification with logistic loss function.
    
    :param n_estimators: `int`. Number of trees to train
    :param learning_rate: `float`. shrinkage parameter. Contribution of each tree to overall estimator
    :param tol: `float`. Tolerance level used in binary classification to convert probabilities to classes
    :param loss: `str`. Default='ls'. Can be one of:
        - 'ls': Regression with Mean Square Error
        - 'lad': Regression with Mean Absolute Error
        - 'logistic': Binary Classification with Logistic loss.
    """

    def __init__(self, n_estimators=100, learning_rate=0.1, loss='ls', tol=0.5):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.tol = tol
        # Base estimator (f_0): Just initialize with 0's
        self.first_estimator = DummyRegressor(strategy='constant', constant=0)
        # Weak learner: Decission tree stumps (single split)
        self.base_estimator = DecisionTreeRegressor(max_depth=1)
        if loss == 'lad':
            self.loss = lad_gradient
        elif loss == 'ls':
            self.loss = mse_gradient
        elif loss == 'logistic':
            self.loss = logistic_gradient
        else:
            raise AttributeError('Unknown loss function: {}'.format(loss))

    def fit(self, X, y):
        """ Fit estimator to data """
        self._estimators = [self.first_estimator]
        # step 0
        f = self.first_estimator.fit(X, y)
        # Begin sequential tree building
        for m in range(self.n_estimators):
            # Step 1: predict using all weak learners trained up to this epoch
            f = self.predict(X)
            # step 2: calculate residuals from previous step
            residuals = self.loss(y, f)
            # step 3: Fit new weak learner g to pseudo-residuals
            g = clone(self.base_estimator).fit(X, residuals)
            # step 4: Store trained weak learner g to be included in prediction at next step
            self._estimators.append(g)
        return self

    def predict(self, X):
        """ Make prediction given feature set. Only use directly for regression """
        # First use the base estimator
        f_0 = self._estimators[0].predict(X)
        # Next add all the individual weak learner's predictions adjusted by the learning rate
        boosting = np.sum([self.learning_rate * f.predict(X) for f in self._estimators[1:]], axis=0)
        return f_0 + boosting


    def _proba_to_class(self, sample):
        """ predict class given a probability """
        return int(sample > self.tol)

    def predict_class(self, X):
        """ Predicts class for binary classification """
        if self.loss != logistic_gradient:
            raise AttributeError('Method only available for binary classification')

        # Turn into probability (in domain 0 to 1)
        predicted_probas = expit(self.predict(X))
        # Now turn probability into class
        return np.array([self._proba_to_class(sample) for sample in predicted_probas])
