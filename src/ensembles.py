import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.tree import DecisionTreeRegressor


class RandomForestMSE:
    def __init__(
        self, n_estimators=100, max_depth=None, feature_subsample_size=None,
        **trees_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.
        max_depth : int
            The maximum depth of the tree. If None then there is no limits.
        feature_subsample_size : float
            The size of feature set for each tree.
            If None then use one-third of all features.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.feature_subsample_size = feature_subsample_size
        self.trees_parameters = trees_parameters
        self.estimators = None
        self.history = None

    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        y : numpy ndarray
            Array of size n_objects
        X_val : numpy ndarray
            Array of size n_val_objects, n_features
        y_val : numpy ndarray
            Array of size n_val_objects
        """

        rng = np.random.default_rng()
        self.history = {}
        self.estimators = []
        max_features = (X.shape[1] // 3 if self.feature_subsample_size is None
                        else self.feature_subsample_size)

        for _ in range(self.n_estimators):
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                max_features=max_features,
                **self.trees_parameters
            )
            idxs = rng.choice(X.shape[0], X.shape[0])
            X_bstrapped = X[idxs, :]
            y_bstrapped = y[idxs]
            self.estimators.append(tree.fit(X_bstrapped, y_bstrapped))
        if X_val is not None and y_val is not None:
            sum_ans = np.zeros(X_val.shape[0], dtype=float)
            for estimator in self.estimators:
                sum_ans += estimator.predict(X_val)
            ans = sum_ans / self.n_estimators
            self.history['rmse'] = np.sqrt(np.mean((y_val - ans) ** 2))
        return self

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        mean_ans = np.zeros(X.shape[0], dtype=float)
        for estimator in self.estimators:
            mean_ans += estimator.predict(X)
        return mean_ans / self.n_estimators


class GradientBoostingMSE:
    def __init__(
            self, n_estimators=100, learning_rate=0.1, max_depth=3,
            feature_subsample_size=None, **trees_parameters):
        """
        n_estimators : int
            The number of trees in the forest.
        learning_rate : float
            Use alpha * learning_rate instead of alpha
        max_depth : int
            The maximum depth of the tree. If None then there is no limits.
        feature_subsample_size : float
            The size of feature set for each tree.
            If None then use one-third of all features.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.feature_subsample_size = feature_subsample_size
        self.trees_parameters = trees_parameters
        self.estimators = None
        self.weights = None
        self.learning_rate = learning_rate
        self.history = None

    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        y : numpy ndarray
            Array of size n_objects
        """
        self.history = {}
        self.estimators = []
        self.weights = np.empty(self.n_estimators + 1, dtype=float)

        self.estimators.append(np.mean(y))
        self.weights[0] = 1
        pred = np.mean(y)
        max_features = (X.shape[1] // 3 if self.feature_subsample_size is None
                        else self.feature_subsample_size)

        for i in range(1, self.n_estimators):
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                max_features=max_features,
                **self.trees_parameters
            )
            self.estimators.append(tree.fit(X, y - pred))
            self.weights[i] = minimize_scalar(
                lambda w: np.sum((y - pred - w * tree.predict(X)) ** 2)).x
            pred += tree.predict(X) * self.learning_rate * self.weights[i]
        if X_val is not None and y_val is not None:
            pred = self.estimators[0]
            for i in range(1, self.n_estimators):
                pred += (self.estimators[i].predict(X_val) *
                         self.learning_rate * self.weights[i])
            self.history['rmse'] = np.sqrt(np.mean((y_val - pred) ** 2))
        return self

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        Returns
        -------
        y : numpy ndarray
        Array of size n_objects
        """
        pred = self.estimators[0]
        for i in range(1, self.n_estimators):
            pred += (self.estimators[i].predict(X) *
                     self.learning_rate * self.weights[i])
        return pred
