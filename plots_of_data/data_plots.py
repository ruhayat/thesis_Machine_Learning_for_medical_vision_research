#-------------------------- Plotting the data ---------------------
#3 different plots:
#- representing the data in 2 dimensions: for each data, computing the mean of the red channel and the mean of the green channel
#- bar graph of the mean of the red channel by image 
#- bar graph of the mean of the green channel by image

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset,Dataset, DataLoader, ConcatDataset
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE




transform_tensor = transforms.Compose([transforms.ToTensor()])
dataset_train = ImageFolder(root = "../data/onh/data_00_onh/training_images/", transform = transform_tensor)
dataset_val = ImageFolder(root = "../data/onh/data_00_onh/val_images/", transform = transform_tensor)
dataset_test = ImageFolder(root = "../data/onh/data_00_onh/test_images/", transform= transform_tensor)
dataset = ConcatDataset([dataset_train, dataset_val, dataset_test])

dataloader  = DataLoader(dataset, batch_size=16, shuffle=True,num_workers=16)

data_red_channel = []
data_green_channel = []
data_mean_red_channel = []
data_mean_green_channel = []
data = []

for i,(inputs,labels) in enumerate(dataloader):
    
    for one_input in inputs:
        data_red_channel.append(one_input[0,:,:].view(-1))
        data_green_channel.append(one_input[1,:,:].view(-1))
        data_mean_red_channel.append(one_input[0,:,:].mean())
        data_mean_green_channel.append(one_input[1,:,:].mean())
        data.append(one_input.view(-1))

    print(i)    






data = torch.stack(data)
data_red_channel = torch.stack(data_red_channel)
data_green_channel = torch.stack(data_green_channel)


plt.figure()

plt.scatter(data_mean_red_channel, data_mean_green_channel)
plt.title('Scatter plot of means per image')
plt.xlabel('Red channel mean')
plt.ylabel('Green channel mean')
plt.savefig('scatter_mean_per_image_after_correction_automated_method.png')


plt.figure()

plt.bar(torch.arange(0,len(data_mean_red_channel)), np.sort(data_mean_red_channel))
plt.title('Bar graph of red channel means per image')
plt.ylabel("Red channel means")
plt.savefig("bar_graph_red_mean_per_image_before_correction.png")


plt.figure()

plt.bar(torch.arange(0,len(data_mean_green_channel)), np.sort(data_mean_green_channel))
plt.title('Bar graph of green channel means per image')
plt.ylabel("Green channel means")
plt.savefig("bar_graph_green_mean_per_image_before_correction.png")


