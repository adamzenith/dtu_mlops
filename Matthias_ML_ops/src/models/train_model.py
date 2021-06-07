
import tqdm
import torch
import argparse
import sys
from model import MyAwesomeModel
from torch import nn
from torch import optim
from tqdm import tqdm
import os
print("Training day and night")
parser = argparse.ArgumentParser(description='Training arguments')
parser.add_argument('--lr', default=0.002,type=float)
parser.add_argument('--epochs',default=10,type=int)
# add any additional argument that you want
args = parser.parse_args()
print(args)
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
# TODO: Implement training loop here

model = MyAwesomeModel()
train_set=torch.load(r'..\..\data\processed\train_FMNIST.pt')

#train_set = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)


print("starting")
model.train()

Loss=[]
for e in tqdm(range(args.epochs),leave=False):
    
    for images, labels in tqdm(train_set,leave=False):

        
        optimizer.zero_grad()
        
        log_ps = model(images)
        loss = criterion(log_ps, labels)
        loss.backward()
        optimizer.step()
        Loss.append(loss.item())
import matplotlib.pyplot as plt
import numpy as np
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

plt.plot(moving_average(Loss,5))
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Loss over time')
plt.savefig(r"..\..\reports\figures\plot_lr"+str(args.lr)+"_epochs"+str(args.epochs)+".png")
torch.save(model.state_dict(), r"..\..\models/checkpoint_lr"+str(args.lr)+"_epochs"+str(args.epochs)+".pth")
