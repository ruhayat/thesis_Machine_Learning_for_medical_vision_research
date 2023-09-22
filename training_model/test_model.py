#-----------------------------apply a model on test data by giving the AUC value and the list of True positive rate and False positive rate
#The True positive reate, the false positive rate are saved in a txt file and the AUC value is saved in another txt file
#Needs to give two parameters : 'onh' or 'periphery' ; '00' or '10' or '20' or '30' or '40' or '50' or '60'


import torch
import time
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pdb
from torchvision.transforms import RandomHorizontalFlip
from torchvision.datasets import ImageFolder
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset, Dataset, random_split, ConcatDataset
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np
from torch.optim.lr_scheduler import StepLR
import torchvision.datasets
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import sys


def calculate_scores(labels, probabilities, threshold):
    
    #calculates true positive, true negative, false positive and false negative

    TP,FP,TN,FN = (0,0,0,0)
    
    

    index_predicted_as_0 = np.where(probabilities<threshold)[0]
    index_predicted_as_1 = np.where(probabilities>=threshold)[0]
   
    

    labels_for_predicted_0 = labels[index_predicted_as_0]
    labels_for_predicted_1 = labels[index_predicted_as_1]

    for label in labels_for_predicted_0:
        if label==0:
            TN+=1
        else:
            FN+=1
        
    for label in labels_for_predicted_1:
        if label==0:
            FP+=1
        else:
            TP+=1

    return TP,FP,TN,FN


def calculate_TPR_FPR(probabilities,labels):
    
    #outputs true positive rate and false negative rate

    liste_TPR = []
    liste_FPR = []

    for threshold in np.arange(0,1,0.1):
        
        TP,FP,TN,FN = calculate_scores(labels, probabilities, threshold)

        TPR = TP/(TP+FN)
        FPR = FP/(FP+TN)

        liste_TPR.append(TPR)
        liste_FPR.append(FPR)

    return liste_TPR,liste_FPR




def copy_green_to_blue(img):
    r,g,b = torch.chunk(img, chunks=3, dim=0)
    return torch.cat((r,g,g), dim=0)



onh_or_periphery = sys.argv[1]
cropping_size = sys.argv[2]


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Lambda(copy_green_to_blue),
    ])

#load the test dataset
test_dataset = ImageFolder(root="../data/"+onh_or_periphery+"/data_"+cropping_size+"_"+onh_or_periphery+"/test_images/Images", transform=transform)
test_dataloader = DataLoader(test_dataset,batch_size=16, shuffle=True, num_workers=16)

len_test_dataset = len(test_dataset)

model  = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
fc_inputs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(fc_inputs, 1024),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(512,256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(128,2)
)

#load the model
loaded_state_dict = torch.load("./models/saved_models_"+cropping_size+"_"+onh_or_periphery+"/model_state_dict.pth")

model.load_state_dict(loaded_state_dict)

model = model.to(device)        
model.eval()

probabilities = []
all_labels = []

with torch.no_grad():
    for i,(inputs,labels) in enumerate(test_dataloader):

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        probabilities.append(F.softmax(outputs,dim=1).cpu()[:,1])
        all_labels.append(labels.cpu())

                
                

    
    probabilities = torch.cat(tuple(probabilities), dim=0)
    
    all_labels = torch.cat(tuple(all_labels),dim=0)
    liste_TPR,liste_FPR = calculate_TPR_FPR(probabilities,all_labels)
                
     




    with open("metrics_test/roc_"+onh_or_periphery+"/roc_metrics.txt","a") as file:
       
        file.write("FPR : ")

        for elem in liste_FPR:
            file.write(str(elem)+" ")

        file.write("\n")
        file.write("TPR : ")

        for elem in liste_TPR:
            file.write(str(elem)+" ")

        file.write("\n")



    with open("metrics_test/auc_"+onh_or_periphery+"/auc.txt","a") as f:
        f.write(cropping_size+" : "+str(auc(liste_FPR, liste_TPR)))
        f.write("\n")
