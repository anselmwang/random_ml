import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import sklearn.utils

class ConstantModel:
    def __init__(self, c):
        self._c = c

    def fit(self, train_mat, train_y):
        pass

    def predict(self, test_mat):
        return np.ones((test_mat.shape[0], 1)) * self._c

def eval(model, name):
    model.fit(train_mat, train_y.reshape(-1, 1))
    pred_y = model.predict(test_mat).flatten()
    plt.plot(test_x, pred_y, label="%s MSE: %.4f" % (name, metrics.mean_squared_error(test_y, pred_y)))

train_x = np.linspace(-10, 10, 200).astype("float32")
train_y = np.sin(train_x).astype("float32")
train_x, train_y = sklearn.utils.shuffle(train_x, train_y)

test_x = np.linspace(-10, 10, 200).astype("float32")
test_y = np.sin(test_x).astype("float32")

train_mat = train_x[:, np.newaxis]
test_mat = test_x[:, np.newaxis]

