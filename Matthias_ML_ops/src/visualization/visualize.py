# - loads a pretrained network, extracts features from the mnist test set (i.e. the features
# 	just before the final classification layer and does [t-SNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)
# 	embedding of the features (color coded according to the class label).
# 	- feel free to create more files/more visualizations (what about investigating/explore the data
# 	distribution of mnist?)
import torch, os, argparse,sys
sys.path.append('../')
from model import MyAwesomeModel
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

#Change directory
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

## Load specified pretrained model
model = MyAwesomeModel()
parser = argparse.ArgumentParser(description='Training arguments')
parser.add_argument('--load_model_from', default="../../models/checkpoint_lr0.002_epochs20.pth")
parser.add_argument('--load_data_from', default="../../data/processed/test.pt")
args = parser.parse_args()
print("Visualizing the features from " + args.load_model_from + " model on " + args.load_data_from + " data")
print("------------------")
state_dict = torch.load(args.load_model_from)
model.load_state_dict(state_dict)

# extract features from the mnist test set and perfom t-SNE embeddings of the features
test_set = torch.load(args.load_data_from)
model.eval()

predictions = torch.empty((64, 64))
labels = torch.empty((64))


## note that we chose to plot only 10 images
i = 0
for images, label in test_set: 
    i += 1
    if images.shape[0] == 64 and i < 30:
        features = model(images, feature=True).detach()
        predictions = torch.cat([predictions, features], dim=0)
        labels = torch.cat([labels,label], dim=0)
    

embedded_features = TSNE(n_components=2).fit_transform(predictions[64:,:]) 
print(embedded_features.shape)
N = 10 # Number of labels

# setup the plot
fig, ax = plt.subplots(1,1, figsize=(6,6))
# define the data
x = embedded_features[:, 0]
y = embedded_features[:, 1]
tag = labels[64:] # Tag each point with a corresponding label    

# define the colormap
cmap = plt.cm.jet
# extract all colors from the .jet map
cmaplist = [cmap(i) for i in range(cmap.N)]
# create the new map
cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

# define the bins and normalize
bounds = np.linspace(0,N,N+1)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

# make the scatter
scat = ax.scatter(x,y,c=tag,cmap=cmap, norm=norm)
# create the colorbar
cb = plt.colorbar(scat, spacing='proportional',ticks=bounds)
cb.set_label('Custom cbar')
ax.set_title('Discrete color mappings')
plt.show()