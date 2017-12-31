import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import sklearn.tree as tree
from approx_sin_common import *


class HighOrderModel:
    def __init__(self, model, order):
        self._model = model
        self._order = order

    def _transform(self, mat):
        assert mat.shape[1] == 1
        new_mat = np.zeros((mat.shape[0], self._order))
        for cur_order in range(self._order):
            new_mat[:, cur_order] = mat[:, 0] ** (cur_order + 1)
        return new_mat

    def fit(self, train_mat, train_y):
        new_mat = self._transform(train_mat)
        self._model.fit(new_mat, train_y)

    def predict(self, test_mat):
        new_mat = self._transform(test_mat)
        return self._model.predict(new_mat)

plt.plot(test_x, test_y, label="original")

eval(ConstantModel(0.), "Const 0")
eval(lm.Ridge(), "Ridge")
eval(HighOrderModel(lm.Ridge(), 1), "Order 1 Ridge")
eval(HighOrderModel(lm.Ridge(), 2), "Order 2 Ridge")
eval(HighOrderModel(lm.Ridge(), 3), "Order 3 Ridge")
eval(tree.DecisionTreeRegressor(max_depth=5), "DT max_depth=5")
eval(tree.DecisionTreeRegressor(), "DT")


plt.legend()