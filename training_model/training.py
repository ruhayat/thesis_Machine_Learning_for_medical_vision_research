#-----------------------------------Training the ResNet-50 model--------------------


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
import sys


def get_mean_and_std_data_distribution(dataloader):
    #not used here, but this function is useful for using data standardization


    mean_R = []
    mean_G = []
    mean_B = []


    for i, (inputs, labels) in enumerate(dataloader):
        
        for image in inputs:
            
            mean_R.append(torch.mean(image[0,:,:]))
            mean_G.append(torch.mean(image[1,:,:]))
            mean_B.append(torch.mean(image[2,:,:]))

    R_mean = np.mean(mean_R)
    G_mean = np.mean(mean_G)
    B_mean = np.mean(mean_B)
    all_means = torch.tensor([R_mean,G_mean,B_mean])
        

    var_R = []
    var_G = []
    var_B = []

    for i, (inputs,labels) in enumerate(dataloader):
        for image in inputs:
            
            image[0] = image[0] - all_means[0]
            image[1] = image[1] - all_means[1]
            image[2] = image[2] - all_means[2]

            image = torch.square(image)
            var_R.append(torch.mean(image[0,:,:]))
            var_G.append(torch.mean(image[1,:,:]))
            var_B.append(torch.mean(image[2,:,:]))
        
    std_R = np.sqrt(np.mean(var_R))
    std_G = np.sqrt(np.mean(var_G))
    std_B = np.sqrt(np.mean(var_B))

    all_std = torch.tensor([std_R,std_G,std_B])

    return all_means, all_std




    
def get_metrics(outputs, targets):
 
    _,predicted_labels = torch.max(outputs, 1)
    accuracy = accuracy_score(targets.cpu(), predicted_labels.cpu())
    precision = precision_score(targets.cpu(), predicted_labels.cpu())
    recall = recall_score(targets.cpu(), predicted_labels.cpu())
    return accuracy, precision, recall




def copy_green_to_blue(img):
    r,g,b = torch.chunk(img, chunks=3, dim=0)
    return torch.cat((r,g,g), dim=0)





cropping_size = sys.argv[2]
onh_or_periphery = sys.argv[1]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)




"""

#CIFAR dataset used for evaluating our model on it
#CIFAR dataset is mutliclass classification but, we only use two class to make it a binary classification

from torchvision.datasets import CIFAR10

# Define a transform to normalize the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Download and load the training data
train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)

test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)

def binary_filter(dataset):
    # Airplane class is 0 and Bird class is 2 in CIFAR-10
    class_indices = [0, 2]
    filtered_data = [(img, label) for img, label in dataset if label in class_indices]
    # Convert labels to binary: 0 (bird) and 1 (airplane)
    binary_data = [(img, 0 if label == 2 else 1) for img, label in filtered_data]
    return binary_data

train_dataset = binary_filter(train_dataset)
test_dataset = binary_filter(test_dataset)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=16)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=16)
val_dataloader = test_dataloader


"""









"""
#simple CNN we use

class model(nn.Module):
    def __init__(self):
        
        super().__init__()

        self.net = nn.Sequential(nn.Conv2d(3,2,4,2,padding=1),
                                nn.ReLU(),
                                nn.BatchNorm2d(2),
                                 
                                nn.Conv2d(2,5,3,2,padding=1),
                                nn.BatchNorm2d(5),
                                nn.ReLU(),
                                 
                                nn.Conv2d(5,9,3,2,padding=1),
                                nn.BatchNorm2d(9),
                                nn.ReLU()
                                )
        
        self.linear1 = nn.Sequential(nn.Linear(7056,55),
                                    nn.ReLU(),
                                    nn.Linear(55,2))
    def forward(self,X):
        

        outputs = self.net(X)
        outputs = torch.flatten(outputs, start_dim=1)
        probabilities = self.linear1(outputs)

      return probabilities


resnet50 = model()
resnet50 = resnet50.to(device)
"""


def training(model,epochs, train_dataloader, val_dataloader, len_train_dataset, len_val_dataset):

    


    # Define Optimizer and Loss Function
    loss_func = nn.CrossEntropyLoss( )

    #used for L2-regularization
    #optimizer = optim.Adam(resnet50.parameters(), lr=0.0001, weight_decay=0.001)
    

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    


    max_loss_val = 100
    nb_not_decreased_loss = 0

    
    for epoch in range(epochs):
    
        print("Epoch: {}/{}".format(epoch + 1, epochs))
        model.train()
       

        running_loss_train = 0.0
        accuracy_train = 0.0
        precision_train = 0.0
        recall_train = 0.0

        for i, (inputs, labels) in enumerate(train_dataloader):
        

            inputs = inputs.to(device)
            labels = labels.to(device)
       

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_func(outputs, labels) 
            loss.backward()
            optimizer.step()
        
            
            running_loss_train += loss.item()

            accuracy,precision,recall = get_metrics(outputs.squeeze(),labels)
        
            accuracy_train += accuracy*inputs.size(0)
            precision_train += precision*inputs.size(0)
            recall_train += recall*inputs.size(0)

            print(
                    "Batch number: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}".format(
                    i, loss.item(), accuracy
                )
            )


        accuracy_train = accuracy_train / len_train_dataset
        running_loss_train = running_loss_train / len_train_dataset
        precision_train = precision_train / len_train_dataset
        recall_train = recall_train / len_train_dataset

        print(f"Epoch {epoch+1}, Loss: {running_loss_train}, Accuracy: {accuracy_train}")
    

        
        
        #validation
        val_accuracy, val_precision, val_recall = 0.0,0.0,0.0
        running_loss_val= 0.0
            
        model.eval()
        for i,(inputs,labels) in enumerate(val_dataloader):
                
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            running_loss_val += loss.item()

                
            accuracy,precision,recall = get_metrics(outputs.squeeze(),labels.float())
            val_accuracy += accuracy * inputs.size(0)
            val_precision += precision * inputs.size(0)
            val_recall += recall*inputs.size(0)
                

        val_accuracy = val_accuracy / len_val_dataset
        val_precision = val_precision / len_val_dataset
        val_recall = val_recall / len_val_dataset
        running_loss_val = running_loss_val / len_val_dataset
           

        #if running loss has not decreased for 10 epochs, we multiply the learning rate by 0.75
        if running_loss_val < max_loss_val:
            max_loss_val = running_loss_val
            nb_not_decreased_loss = 0
        else:
            nb_not_decreased_loss += 1
                
        if nb_not_decreased_loss == 10:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.75
                
            nb_not_decreased_loss = 0
                
        
        print(f"Validation loss: {running_loss_val} accuracy: {val_accuracy}, precision: {val_precision}, recall: {val_recall}")
       
        with open("metrics_validation/metrics_validation_"+cropping_size+"_"+onh_or_periphery+".txt","a") as f:
            f.write("epoch "+str(epoch+1)+" => Training loss : "+str(running_loss_train)+" accuracy : "+str(accuracy_train)+", precision : "+str(precision_train)+", recall : "+str(recall_train))
            f.write("\n")
            f.write("epoch "+str(epoch+1)+" => Validation loss : "+str(running_loss_val)+" accuracy : "+str(val_accuracy)+", precision : "+str(val_precision)+ ", recall : "+ str(val_recall))
            f.write("\n") 
        torch.save(model.state_dict(), "./models/saved_models_"+cropping_size+"_"+onh_or_periphery+"/model_state_dict_"+str(epoch+1)+".pth")
        





if __name__ == '__main__':

    

    

    transform_train = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Lambda(copy_green_to_blue),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomHorizontalFlip(p=0.5)
    ])

    transform_val = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Lambda(copy_green_to_blue),
    ])


    epochs = 40
    batch_size = 16
    


    train_dataset = ImageFolder(root="../data/"+onh_or_periphery+"/data_"+cropping_size+"_"+onh_or_periphery+"/training_images/Images", transform = transform_train)
    val_dataset = ImageFolder(root="../data/"+onh_or_periphery+"/data_"+cropping_size+"_"+onh_or_periphery+"/val_images/Images", transform=transform_val)
    test_dataset = ImageFolder(root="../data/"+onh_or_periphery+"/data_"+cropping_size+"_"+onh_or_periphery+"/test_images/Images", transform = transform_val)



    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers = 16)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=16)


    
    
    resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    for param in resnet50.parameters():
       param.requires_grad = False


    # Unfreeze some layers

    #for param in resnet50.layer1.parameters():
    #    param.requires_grad = True

    #for param in resnet50.layer2.parameters():
    #    param.requires_grad = True

    #for param in resnet50.layer3.parameters():
    #    param.requires_grad = True
    
    #for param in resnet50.layer4.parameters():
    #    param.requires_grad = True
    


    fc_inputs = resnet50.fc.in_features
    resnet50.fc = nn.Sequential(
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
    
    
    # Convert model to be used on GPU
    resnet50 = resnet50.to(device)


    len_train_dataset = len(train_dataset)
    len_val_dataset = len(val_dataset)
    len_test_dataset = len(test_dataset)

    print("len training dataset ",len_train_dataset)
    print("len validation dataset ",len_val_dataset)
     
    training(resnet50, epochs, train_dataloader, val_dataloader, len_train_dataset, len_val_dataset)
   







