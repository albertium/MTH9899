import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor


def generate_test_data(n: int):
    x = np.random.randn(n, 6)
    x[:, 5] = np.where(x[:, 5] > 0, 1, 0)
    y = np.where(x[:, 0] > 0, 2, 5)
    y = y + np.where(x[:, 1] > 0, -3, 3)
    y = y + np.where(x[:, 2] > 0, 0, 0.5)
    y[np.where(x[:, 5] == 1)[0]] = y[np.where(x[:, 5] == 1)[0]] + 0.5 + np.random.randn(len(np.where(x[:, 5])[0] == 1))/10
    y = y + np.random.randn(n) * 3
    return x, y


def calculate_rsq(y, y_hat):
    return 1 - np.var(y - y_hat) / np.var(y)


class TreeNode:
    def predict(self, x) -> np.ndarray:
        assert False

    def depth(self):
        assert False


class BranchNode(TreeNode):
    def __init__(self, left, right, split_var_index, split_var_value):
        self.left = left
        self.right = right
        self.split_var_index = split_var_index
        self.split_var_value = split_var_value

    def predict(self, x):
        svar = x[:, self.split_var_index]
        is_left = svar < self.split_var_value
        leftx = x[is_left]
        rightx = x[~is_left]

        rv = np.zeros(x.shape[0])
        rv[is_left] = self.left.predict(leftx)
        rv[~is_left] = self.right.predict(rightx)

        return rv

    def depth(self):
        return 1 + max(self.left.depth(), self.right.depth())


class LeafNode(TreeNode):
    def __init__(self, mu):
        self.mu = mu

    def predict(self, x):
        return np.repeat(self.mu, x.shape[0])

    def depth(self):
        return 1


class RegressionTree:
    def __init__(self, max_depth: int, min_points_in_leaf: int):
        self.max_depth = max_depth
        self.min_points_in_leaf = min_points_in_leaf
        self.fitted = False
        self.root = TreeNode()

    def predict(self, x):
        assert self.fitted
        return self.root.predict(x)

    def fit(self, x, y):
        self.fitted = True
        self.root = self.fit_internal(x, y, 1)

    def fit_internal(self, x: np.ndarray, y: np.ndarray, current_depth: int) -> TreeNode:
        # implement this
        num_features = x.shape[1]
        num_rows = x.shape[0]
        var_orig = np.var(y)

        if current_depth >= self.max_depth:
            return LeafNode(np.mean(y))

        best_variable = None
        best_var = var_orig
        best_threshold = None
        best_ind = None

        # Here, we have to loop over all features and figure out which one
        # might be splittable, and if it is, how to split it to maximize Variance Reduction
        for i in range(num_features):
            # a lot of code goes here
            feature = x[:, i]
            cuts = np.linspace(0, 1, min(100, len(np.unique(feature))) + 1)[1: -1]
            bins = np.percentile(feature, cuts * 100)

            for threshold in bins:
                ind = feature < threshold
                subset1, subset2 = y[ind], y[~ind]
                if subset1.shape[0] < self.min_points_in_leaf or subset2.shape[0] < self.min_points_in_leaf:
                    continue
                new_var = (np.var(subset1) * subset1.shape[0] + np.var(subset2) * subset2.shape[0]) / num_rows
                if new_var < best_var:
                    best_var = new_var
                    best_variable = i
                    best_threshold = threshold
                    best_ind = ind

        if best_variable is None:
            return LeafNode(np.mean(y))
        else:
            left = self.fit_internal(x[best_ind], y[best_ind], current_depth + 1)
            right = self.fit_internal(x[~best_ind], y[~best_ind], current_depth + 1)
            return BranchNode(left, right, best_variable, best_threshold)

    def depth(self):
        return self.root.depth()


if __name__ == '__main__':
    # np.random.seed(123)
    x_train, y_train = generate_test_data(50000)
    x_test, y_test = generate_test_data(50000)

    depths = list(range(1, 20, 2))
    rsq_train = []
    rsq_test = []
    for depth in depths:
        tree = RegressionTree(max_depth=depth, min_points_in_leaf=30)
        # tree = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=30)

        # training performance
        tree.fit(x_train, y_train)
        y_hat = tree.predict(x_train)
        rsq_train.append(calculate_rsq(y_train, y_hat))

        # test performance
        y_hat = tree.predict(x_test)
        rsq_test.append(calculate_rsq(y_test, y_hat))

    print(rsq_train)
    print(rsq_test)
    plt.plot(depths, rsq_train, marker='x', label='Training')
    plt.plot(depths, rsq_test, marker='.', label='Test')
    plt.legend()
    plt.show()