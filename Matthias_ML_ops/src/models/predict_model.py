import tqdm
import torch
import argparse
import sys
from model import MyAwesomeModel
from torch import nn

# from torch import optim
from tqdm import tqdm
from PIL import Image
import os
from torchvision import transforms

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)


parser = argparse.ArgumentParser(description="Training arguments")
parser.add_argument("--load_model_from", default=r"..\..\models\checkpoint_lr0.002_epochs10.pth")
parser.add_argument("--file", default=r"..\..\data\raw\00000.jpg")
# add any additional argument that you want
args = parser.parse_args()

# TODO: Implement evaluation logic here
model = MyAwesomeModel()
criterion = nn.NLLLoss()

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
if args.load_model_from:
    model.load_state_dict(torch.load((args.load_model_from)))
pic = Image.open(args.file)
pic = transform(pic)
import matplotlib.pyplot as plt
import numpy as np
from torch import nn, optim
from torch.autograd import Variable


def view_classify(img, ps, version="MNIST"):
    """Function for viewing an image and it's predicted classes."""
    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6, 9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis("off")
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    if version == "MNIST":
        ax2.set_yticklabels(np.arange(10))
    elif version == "Fashion":
        ax2.set_yticklabels(
            [
                "T-shirt/top",
                "Trouser",
                "Pullover",
                "Dress",
                "Coat",
                "Sandal",
                "Shirt",
                "Sneaker",
                "Bag",
                "Ankle Boot",
            ],
            size="small",
        )
    ax2.set_title("Class Probability")
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()
    plt.show()


with torch.no_grad():
    logps = model(pic)

# Output of the network are log-probabilities, need to take exponential for probabilities
ps = torch.exp(logps)
view_classify(pic, ps)
