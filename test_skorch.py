import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import skorch
from skorch.dataset import CVSplit
from skorch.net import *
import itertools

class Net(nn.Module):
    def __init__(self, num_input):
        super(Net, self).__init__()
        self.dense0 = nn.Linear(num_input, 100)

        for i in range(len(NUM_CLASS_LIST)):
            setattr(self, "output_%s" % i, nn.Linear(100, NUM_CLASS_LIST[i]))

    def forward(self, x):
        x = F.relu(self.dense0(x))

        output_vars = []
        for i in range(len(NUM_CLASS_LIST)):
            output_vars.append(F.softmax(getattr(self, "output_%s" % i)(x)))

        x = torch.cat(output_vars, dim=1)
        return x

class MyEpochScoring(skorch.callbacks.EpochScoring):
    # inherit to remove row "y_test = self.target_extractor(y_test)" in on_epoch_end
    # and to override _scoring function
    def _scoring(self, net, X_test, y_test):
        return self.scoring(net, X_test, y_test)

    def on_epoch_end(
            self,
            net,
            X,
            y,
            X_valid,
            y_valid,
            **kwargs):
        history = net.history

        if self.on_train:
            X_test, y_test = X, y
        else:
            X_test, y_test = X_valid, y_valid

        if X_test is None:
            return

        current_score = self._scoring(net, X_test, y_test)
        history.record(self.name_, current_score)

        is_best = self._is_best_score(current_score)
        if is_best is None:
            return

        history.record(self.name_ + '_best', is_best)
        if is_best:
            self.best_score_ = current_score


class MultipleOutputNNClassifier(skorch.net.NeuralNetClassifier):
    def get_default_callbacks(self):
        return [
            ('epoch_timer', EpochTimer()),
            ('train_loss', MyEpochScoring(
                self.my_score,
                name='train_loss',
                on_train=True,
            )),
            ('valid_loss', MyEpochScoring(
                self.my_score,
                name='valid_loss',
            )),
            ('print_log', PrintLog()),]

    def my_score(self, net, X, y):
        y_pred = net.predict_proba(X)
        return self._score(Variable(torch.FloatTensor(y_pred)), y).data[0]

    def _score(self, y_pred, y_true):
        mask = y_true["mask"]
        if isinstance(mask, np.ndarray):
            mask = torch.IntTensor(mask)
        y_true = to_var(y_true["label"], use_cuda=self.use_cuda)
        y_pred_log = torch.log(y_pred)

        losses = []
        for i in range(len(NUM_CLASS_LIST)):
            start, end = SEG_LOOKUP[i], SEG_LOOKUP[i + 1]
            y_pred_log_one_task = y_pred_log[:, start:end]
            y_pred_log_one_task_masked = y_pred_log_one_task[
                mask[:, i].unsqueeze(1).expand_as(y_pred_log_one_task).byte()]
            y_pred_log_one_task_masked = y_pred_log_one_task_masked.view(int(len(y_pred_log_one_task_masked)/ y_pred_log_one_task.shape[1]),
                                                                         y_pred_log_one_task.shape[1])

            y_true_one_task = y_true[:, i]
            y_true_one_task_masked = y_true_one_task[mask[:, i].byte()]

            losses.append(self.criterion_(y_pred_log_one_task_masked, y_true_one_task_masked))

        return sum(losses)

    def get_loss(self, y_pred, y_true, X=None, training=False):
        return self._score(y_pred, y_true)


NUM_SAMPLE = 1000
NUM_CLASS_LIST = [4, 4, 4]
SEG_LOOKUP = [0] + list(itertools.accumulate(NUM_CLASS_LIST))
NUM_FEAT = 10

net = MultipleOutputNNClassifier(Net,
                                 module__num_input = NUM_FEAT,
                                 train_split=CVSplit(5, stratified=False),
                                 max_epochs=100,
                                 )

X = np.random.random((NUM_SAMPLE, NUM_FEAT)).astype(np.float32)

y = {
    "label": np.random.randint(0, 4, size=(NUM_SAMPLE, len(NUM_CLASS_LIST))).astype("int64"),
    "mask": np.random.randint(0, 2, size=(NUM_SAMPLE, len(NUM_CLASS_LIST))),
}

net.fit(X, y)