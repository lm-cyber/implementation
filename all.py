from sklearn.metrics import mean_squared_error
import numpy as np

class MyGradientLinearRegression():
    def __init__(self, fit_intercept):
        self.fit_intercept = fit_intercept
        self.w = None

    def predict(self, X):
        n, k = X.shape
        if self.fit_intercept:
            X_train = np.hstack((X, np.ones((n, 1))))

        y_pred = X_train @ self.w

        return y_pred

    def get_weights(self):
        return self.w

    def fit(self, X, y, lr=0.01, max_iter=100):

        n, k = X.shape

        # случайно инициализируем веса
        if self.w is None:
            self.w = np.random.randn(k + 1 if self.fit_intercept else k)

        X_train = np.hstack((X, np.ones((n, 1)))) if self.fit_intercept else X

        self.losses = []

        for iter_num in range(max_iter):
            y_pred = self.predict(X)
            self.losses.append(mean_squared_error(y_pred, y))

            grad = self._calc_gradient(X_train, y, y_pred)

            assert grad.shape == self.w.shape, f"gradient shape {grad.shape} is not equal weight shape {self.w.shape}"
            self.w -= lr * grad

        return self

    def _calc_gradient(self, X, y, y_pred):
        grad = 2 * (y_pred - y)[:, np.newaxis] * X
        grad = grad.mean(axis=0)
        return grad

    def get_losses(self):
        return self.losses


class KNN:
    def __init__(self, n_neighbors: int = 3, metric: str = 'cosinus', coeff: int = 0.9) -> None:
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.coeff = coeff

    def get_class(self, classes: np.array) -> int:
        unique = np.unique(classes)
        targets = {}

        # используем веса
        for i in range(len(unique)):
            targets[unique[i]] = 0
            for j in range(self.n_neighbors):
                w = self.coeff ** (j)
                targets[unique[i]] += int(unique[i] == classes[j]) * w

        class_index = list(targets.values()).index(max(targets.values()))
        return list(targets.keys())[class_index]

    def fit(self, features: np.array, target: np.array):
        self.features = features
        self.target = target

    def predict(self, features: np.array):

        # косинусная мера
        if self.metric == 'cosinus':
            result_matrix = (features @ self.features.T)  # матрица скалярных произведений
            features_norm = ((features @ features.T) ** 0.5).diagonal()  # нормы векторов в матрице features
            self_norm = ((self.features @ self.features.T) ** 0.5).diagonal()  # нормы векторов в матрице self.features

            result_matrix = result_matrix / np.array([features_norm] * (self.features.shape[0])).T / (
                np.array([self_norm] * features.shape[0]))
            classes = self.target[
                np.flip(np.argsort(result_matrix, axis=1, kind='mergesort')[:, -self.n_neighbors:], axis=1)]

            result = np.zeros(features.shape[0])

            for i in range(features.shape[0]):
                result[i] = self.get_class(classes[i])

            return result

        else:
            result_matrix = np.zeros((features.shape[0], self.features.shape[0]))
            for i in range(result_matrix.shape[0]):
                if self.metric == 'manhattan':
                    result_matrix[i] = np.sum(np.abs(self.features - features[i]), axis=-1)
                elif self.metric == 'euclidien':
                    result_matrix[i] = (np.sum((self.features - features[i]) ** 2, axis=-1)) ** 0.5

            classes = self.target[np.argsort(result_matrix, axis=1)[:, :self.n_neighbors]]

            result = np.zeros(result_matrix.shape[0])

            for i in range(result_matrix.shape[0]):
                result[i] = self.get_class(classes[i])

            return result




def logit(x, w):
    return np.dot(x, w)


def sigmoid(h):
    return 1. / (1 + np.exp(-h))


def generate_batches(X, y, batch_size):
    assert len(X) == len(y)
    np.random.seed(42)
    X = np.array(X)
    y = np.array(y)
    perm = np.random.permutation(len(X))

    for batch_start in range(0, len(X) - batch_size + 1, batch_size):
        prog = perm[batch_start:batch_start + batch_size]
        yield X[prog], y[prog]
class MyLogisticRegression(object):
    def __init__(self):
        self.w = None

    def fit(self, X, y, epochs=10, lr=0.1, batch_size=100):
        n, k = X.shape
        if self.w is None:
            np.random.seed(42)
            self.w = np.random.randn(k + 1)

        X_train = np.concatenate((np.ones((n, 1)), X), axis=1)
        losses = []

        for i in range(epochs):
            for X_batch, y_batch in generate_batches(X_train, y, batch_size):
                predictions = self._predict_proba_internal(X_batch)
                loss = self.__loss(y_batch, predictions)
                losses.append(loss)
                self.w -= lr * self.get_grad(X_batch, y_batch, predictions)

        return losses

    def get_grad(self, X_batch, y_batch, predictions):

        grad_basic = X_batch.T @ (predictions - y_batch)
        return grad_basic

    def predict_proba(self, X):
        n, k = X.shape
        X_ = np.concatenate((np.ones((n, 1)), X), axis=1)
        return sigmoid(logit(X_, self.w))

    def _predict_proba_internal(self, X):
        return sigmoid(logit(X, self.w))

    def predict(self, X, threshold=0.5):
        return self.predict_proba(X) >= threshold

    def get_weights(self):
        return self.w.copy()

    def __loss(self, y, p):
        p = np.clip(p, 1e-10, 1 - 1e-10)
        return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))



# not my
# source
# https://gist.github.com/BenjaminFraser/e372611002a963d0f7d8dc4b30fcb44b
class DecisionTree():
    """ Form a basic decision tree """

    def __init__(self, x, y, idxs=None, oob_idxs=None,
                 min_leaf=5, feat_proportion=1.0):
        if idxs is None:
            idxs = np.arange(len(y))
        self.x = x
        self.y = y
        self.idxs = idxs
        self.oob_idxs = oob_idxs
        self.min_leaf = min_leaf
        self.feat_proportion = feat_proportion
        self.rows = len(idxs)
        self.cols = self.x.shape[1]
        self.val = np.mean(y[idxs])
        self.score = float('inf')
        self.binary_split()

    def __repr__(self):
        """ String reputation of our decision tree """
        text = f'n: {self.rows}, val: {self.val}'
        if not self.is_leaf:
            text += f', score: {self.score}, split: {self.split}, var: {self.split_name}'
        return text

    def binary_split(self):
        """ find best feature and level to split at to produce greatest
            reduction in variance """

        # randomly select sub-sample of features
        num_feat = int(np.ceil(self.cols * self.feat_proportion))
        col_idxs = range(self.cols)
        feature_subset = np.random.permutation(col_idxs)[:num_feat]

        # iteratively split each col and find best
        for i in feature_subset:
            self.best_binary_split(i)
        # if leaf node stop
        if self.score == float('inf'):
            return

        # get split col and idxs for lhs and rhs splits
        x = self.split_col_values
        lhs = np.nonzero(x <= self.split)[0]
        rhs = np.nonzero(x > self.split)[0]

        # create new decision trees for each split
        self.left_split = DecisionTree(self.x, self.y, self.idxs[lhs])
        self.right_split = DecisionTree(self.x, self.y, self.idxs[rhs])

    def best_binary_split(self, feat_idx):
        """ Find best binary split for the given feature """
        x = self.x.values[self.idxs, feat_idx]
        y = self.y[self.idxs]

        # sort our data
        sorted_ind = np.argsort(x)
        sorted_x = x[sorted_ind]
        sorted_y = y[sorted_ind]

        # get count, sum and square sum of lhs and rhs
        lhs_count = 0
        rhs_count = self.rows
        lhs_sum = 0.0
        rhs_sum = sorted_y.sum()
        lhs_sum2 = 0.0
        rhs_sum2 = np.square(sorted_y).sum()

        # iterate through all values of selected feature - eval score
        for i in range(0, self.rows - self.min_leaf):
            x_i = sorted_x[i]
            y_i = sorted_y[i]

            # update count and sums
            lhs_count += 1
            rhs_count -= 1
            lhs_sum += y_i
            rhs_sum -= y_i
            lhs_sum2 += y_i ** 2
            rhs_sum2 -= y_i ** 2

            # if less than min leaf or dup value - skip
            if i < self.min_leaf - 1 or x_i == sorted_x[i + 1]:
                continue

            # find standard deviations of left and right sides
            lhs_std = self.standard_deviation(lhs_count, lhs_sum, lhs_sum2)
            rhs_std = self.standard_deviation(rhs_count, rhs_sum, rhs_sum2)

            # find weighted score
            current_score = (lhs_count * lhs_std) + (rhs_count * rhs_std)

            # if score lower (better) than previous, update
            if current_score < self.score:
                self.feat_idx = feat_idx
                self.score = current_score
                self.split = x_i

    def standard_deviation(self, n, summed_vals, summed_vals_squared):
        """ Standard deviation using summed vals, sum of squares, and data size """
        return np.sqrt((summed_vals_squared / n) - np.square(summed_vals / n))

    def predict(self, x):
        """ Find and return predictions for all the samples in x """
        return np.array([self.predict_sample(x_i) for x_i in x])

    def predict_sample(self, x_i):
        """ Take a sample x_i and return the predicted value using recursion """
        # if leaf node - return mean value
        if self.is_leaf:
            return self.val

        # if value less than tree split value lhs, else rhs
        elif x_i[self.feat_idx] <= self.split:
            tree = self.left_split
        else:
            tree = self.right_split

        # recursively continue through the tree with x_i until leaf node
        return tree.predict_sample(x_i)

    @property
    def split_name(self):
        """ return name of column we are splitting on """
        return self.x.columns[self.feat_idx]

    @property
    def split_col_values(self):
        """ return values of column we have split on """
        return self.x.values[self.idxs, self.feat_idx]

    @property
    def is_leaf(self):
        """ If leaf node, score will be infinity """
        return self.score == float('inf')