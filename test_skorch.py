import numpy as np
import torch
import torch.nn.functional as F
import skorch
from skorch.dataset import CVSplit

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.dense_a = torch.nn.Linear(10, 100)
        self.dense_b = torch.nn.Linear(20, 100)
        self.output = torch.nn.Linear(200, 2)

    def forward(self, key_a, key_b):
        hid_a = F.relu(self.dense_a(key_a))
        hid_b = F.relu(self.dense_b(key_b))
        concat = torch.cat((hid_a, hid_b), dim=1)
        out = F.softmax(self.output(concat))
        return out

net = skorch.NeuralNetClassifier(MyModule, train_split=CVSplit(5, stratified=False),)

X = {
    'key_a': np.random.random((1000, 10)).astype(np.float32),
    'key_b': np.random.random((1000, 20)).astype(np.float32),
}
y = np.random.randint(0, 2, size=1000)

y = {
    "y_a": np.random.randint(0, 2, size=1000),
    "y_b": np.random.randint(0, 2, size=1000),
}


net.fit(X, y)