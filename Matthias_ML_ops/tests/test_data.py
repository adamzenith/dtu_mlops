import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
sys.path.append(r"..")
print(sys.path)
from src.data.make_MNIST import create_MNIST

import torch

print(os.getcwd())


def test_data():

    datatest, datatrain = create_MNIST()
    assert len(datatrain) == 60000
    assert len(datatest) == 10000
    assert datatrain[0][0].shape == torch.Size([1, 28, 28])
    assert datatest[0][0].shape == torch.Size([1, 28, 28])
    # assert that each datapoint has shape [1,28,28] or [728] depending on how you choose to format
    # assert that all labels are represented
