#---------------------------------Cleaning first method------------------------


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
from PIL import Image
import os
from cleanlab.outlier import OutOfDistribution
from cleanlab.rank import find_top_issues
import timm
import torchvision

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")





def get_retained_filenames(dataloader, dataset):

    liste_retained_filenames = []

    for i, (inputs,labels) in enumerate(dataloader):
        mean_red = inputs[0,0,:,:].mean()
        mean_green = inputs[0,1,:,:].mean()

        #threshold
        if mean_red > 0.1 and mean_green > 0.1 and mean_red < 0.5 and mean_green < 0.4 and (mean_red - mean_green) < 0.25 and (mean_red - mean_green) > 0.05 :
            liste_retained_filenames.append(dataset.imgs[i][0][-41:])

    return liste_retained_filenames



def remove_uncorrect_png_files(data_dir, liste_retained_filenames):
    filenames = os.listdir(data_dir)
    
    for filename in filenames:
        if filename not in liste_retained_filenames:
            os.system("mv "+data_dir+"/"+filename+ " /home/rhayat_mehresearch_org/data/onh/data_00_onh/Images/ ")
        
    

transform_tensor = transforms.Compose([transforms.ToTensor()])


#We first collect all the data
root_dir_data = "../data/onh/data_00_onh/Images"
dataset = ImageFolder(root_dir_data,transform = transform_tensor)
dataloader = DataLoader(dataset,batch_size=1,num_workers=16)

#we get the PNG filenames of the data that have been retained
liste_retained_filenames = get_retained_filenames(dataloader, dataset)

#The, we remove all the data that have not been retained
#Only the cleaned data remain stored in the folder data/
remove_uncorrect_png_files("/home/rhayat_mehresearch_org/data/onh/data_00_onh/Images/data/",liste_retained_filenames)

