import tqdm
import torch
import argparse
import sys
from model import MyAwesomeModel
from torch import nn

# from torch import optim
from tqdm import tqdm
import os

print("Evaluating until hitting the ceiling")
parser = argparse.ArgumentParser(description="Training arguments")
parser.add_argument("--load_model_from", default=r"C:\Users\matth\Documents\GitHub\dtu_mlops\Matthias_ML_ops\models\checkpoint_lr0.002_epochs10.pth")
# add any additional argument that you want
args = parser.parse_args()
print(args)
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
# TODO: Implement evaluation logic here
model = MyAwesomeModel()
criterion = nn.NLLLoss()
if args.load_model_from:
    model.load_state_dict(torch.load(args.load_model_from))
test_set = torch.load(r"..\..\data\processed\test_FMNIST.pt")


accuracies = []
with torch.no_grad():
    model.eval()
    for images, labels in tqdm(test_set):
        ## TODO: Implement the validation pass and print out the validation accuracy

        ps = torch.exp(model(images))
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy = torch.mean(equals.type(torch.FloatTensor))
        accuracies.append(accuracy)
        # print(f'Accuracy: {accuracy.item()*100}%')
accuracy = sum(accuracies) / len(accuracies)
print(accuracy)
