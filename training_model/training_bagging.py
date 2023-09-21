#---------------------------implementation of a bagging method------------------------

#5 ResNet-50 classifiers with the same set of last fully-connected layers are trained and evaluated, and the majority of predicted labels is the predicted label



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



def get_mean_and_std_data_distribution(dataloader):
    
    #not used here, but useful for data standardization

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




    

#applying a transformation on a pytorch dataset
class MyDataset(torch.utils.data.Dataset):
        def __init__(self, subset, transform=None):
            self.subset = subset
            self.transform = transform
                                    
        def __getitem__(self, index):
            x, y = self.subset[index]
            if self.transform:
                x = self.transform(x)
            return x, y
                                                                                        
        def __len__(self):
            return len(self.subset)


def copy_green_to_blue(img):
    r,g,b = torch.chunk(img, chunks=3, dim=0)
    return torch.cat((r,g,g), dim=0)






cropping_size = "00"
onh_or_periphery = "onh"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)



def get_metrics(outputs, targets):
 
    _,predicted_labels = torch.max(outputs, 1)
    accuracy = accuracy_score(targets.cpu(), predicted_labels.cpu())
    precision = precision_score(targets.cpu(), predicted_labels.cpu())
    recall = recall_score(targets.cpu(), predicted_labels.cpu())
    return accuracy, precision, recall




def evaluate_bagging(models,dataloader, len_dataset):
    
    total_accuracy, total_precision, total_recall = 0.0,0.0,0.0
            
    models[0].eval()
    models[1].eval()
    models[2].eval()
    models[3].eval()
    models[4].eval()

    
    for i,(inputs,labels) in enumerate(dataloader):
                

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs1 = models[0](inputs)
        outputs2 = models[1](inputs)
        outputs3 = models[2](inputs)
        outputs4 = models[3](inputs)
        outputs5 = models[4](inputs)

                
        _,predicted_labels1 = torch.max(outputs1, 1)
        _,predicted_labels2 = torch.max(outputs2, 1)
        _,predicted_labels3 = torch.max(outputs3, 1)
        _,predicted_labels4 = torch.max(outputs4, 1)
        _,predicted_labels5 = torch.max(outputs5, 1)

        #We take the majority
        predicted_labels = []
        for i in range(len(predicted_labels1)):
            preds_i = [predicted_labels1[i], predicted_labels2[i], predicted_labels3[i], predicted_labels4[i], predicted_labels5[i]]                    
            if sum(preds_i) > (len(preds_i) / 2):
                predicted_labels.append(1)
            else:
                predicted_labels.append(0)


        accuracy = accuracy_score(labels.cpu(), predicted_labels)
        precision = precision_score(labels.cpu(), predicted_labels)
        recall = recall_score(labels.cpu(), predicted_labels)
                

        total_accuracy += accuracy * inputs.size(0)
        total_precision += precision * inputs.size(0)
        total_recall += recall * inputs.size(0)

    total_accuracy = total_accuracy / len_dataset
    total_precision = total_precision / len_dataset
    total_recall = total_recall / len_dataset
    
    return total_accuracy, total_precision, total_recall



def training(models, epochs, sub_train_dataloaders, train_dataloader, val_dataloader, len_sub_train_dataset, len_train_dataset, len_val_dataset):

    


    loss_func1 = nn.CrossEntropyLoss( )
    loss_func2 = nn.CrossEntropyLoss( )
    loss_func3 = nn.CrossEntropyLoss( )
    loss_func4 = nn.CrossEntropyLoss( )
    loss_func5 = nn.CrossEntropyLoss( )
    
    optimizer1 = optim.Adam(models[0].parameters(), lr=0.0001)
    optimizer2 = optim.Adam(models[1].parameters(), lr=0.0001)
    optimizer3 = optim.Adam(models[2].parameters(), lr=0.0001)
    optimizer4 = optim.Adam(models[3].parameters(), lr=0.0001)
    optimizer5 = optim.Adam(models[4].parameters(), lr=0.0001)
    
    losses = [loss_func1, loss_func2, loss_func3, loss_func4, loss_func5]
    optimizers = [optimizer1, optimizer2, optimizer3, optimizer4, optimizer5]


    max_loss_val = 100
    nb_not_decreased_loss = 0


    
    for epoch in range(epochs):

        print("Epoch: {}/{}".format(epoch + 1, epochs))

        for j in range(5):

            print("Model n°",j+1)
    
            model = models[j]
            model.train()

            sub_running_loss_train = 0.0
            sub_accuracy_train = 0.0
            sub_precision_train = 0.0
            sub_recall_train = 0.0
            
            loss_func = losses[j]
            optimizer = optimizers[j]
            sub_train_dataloader = sub_train_dataloaders[j]


            for i, (inputs, labels) in enumerate(sub_train_dataloader):
        

                inputs = inputs.to(device)
                labels = labels.to(device)
       

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_func(outputs, labels) 
                loss.backward()
                optimizer.step()
        
            
                sub_running_loss_train += loss.item()

                accuracy,precision,recall = get_metrics(outputs.squeeze(),labels)
        
                sub_accuracy_train += accuracy*inputs.size(0)
                sub_precision_train += precision*inputs.size(0)
                sub_recall_train += recall*inputs.size(0)

                print(
                        "Batch number: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}".format(
                        i, loss.item(), accuracy
                    )
                )
                
            if (epoch+1)%10==0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.1

            


            sub_accuracy_train = sub_accuracy_train / len_sub_train_dataset
            sub_running_loss_train = sub_running_loss_train / len_sub_train_dataset
            sub_precision_train = sub_precision_train / len_sub_train_dataset
            sub_recall_train = sub_recall_train / len_sub_train_dataset

            print(f"Epoch {epoch+1}, Loss: {sub_running_loss_train}, Accuracy: {sub_accuracy_train}, Precision: {sub_precision_train}, Recall: {sub_recall_train}")
        
        

        accuracy_train, precision_train, recall_train = evaluate_bagging(models, train_dataloader, len_train_dataset)    
        
        print("--------------------------------------------------------------------")
        print(f"Train accuracy: {accuracy_train}, precision: {precision_train}, recall: {recall_train}") 
        print("--------------------------------------------------------------------")
        
        #validation
            
        val_accuracy, val_precision, val_recall = evaluate_bagging(models, val_dataloader, len_val_dataset)

        print("----------------------------------------------------")
        print(f"Validation accuracy: {val_accuracy}, precision: {val_precision}, recall: {val_recall}")
        print("----------------------------------------------------")
        

        with open("metrics_validation/metrics_validation_"+cropping_size+"_"+onh_or_periphery+"_fine_tuning/metrics_validation_"+cropping_size+"_"+onh_or_periphery+"_4_blocks_unfrozen_bagging.txt","a") as f:
            f.write("epoch "+str(epoch+1)+" => Training accuracy : "+str(accuracy_train)+", precision : "+str(precision_train)+", recall : "+str(recall_train))
            f.write("\n")
            f.write("epoch "+str(epoch+1)+" => Validation accuracy : "+str(val_accuracy)+", precision : "+str(val_precision)+ ", recall : "+ str(val_recall))
            f.write("\n") 
        #torch.save(model.state_dict(), "./models/saved_models_"+cropping_size+"_"+onh_or_periphery+"/model_state_dict_"+str(epoch+1)+".pth")
        

        


    




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
        transforms.Lambda(copy_green_to_blue)
    ])

    epochs = 40
    batch_size = 16
    


    train_dataset = ImageFolder(root="../data/onh/data_"+cropping_size+"_"+onh_or_periphery+"/training_images/Images")
    val_dataset = ImageFolder(root="../data/onh/data_"+cropping_size+"_"+onh_or_periphery+"/val_images/Images", transform=transform_val)
    test_dataset = ImageFolder(root="../data/onh/data_"+cropping_size+"_"+onh_or_periphery+"/test_images/Images", transform = transform_val)

    

    len_train_dataset = len(train_dataset)
    len_val_dataset = len(val_dataset)
    len_test_dataset = len(test_dataset)
   
    print("len training dataset ",len_train_dataset)
    print("len validation dataset ",len_val_dataset)
    

    train_dataloader = DataLoader(MyDataset(train_dataset, transform_train), batch_size=batch_size, shuffle=True, num_workers = 16)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=16)


    samples_size = int(0.6*len(train_dataset)) #size of each subsample of training data

    train_dataloaders = []    
    list_models = []

    for i in range(5): #5 classifiers to train

        indices = torch.randperm(len(train_dataset))[:samples_size]
        
        subset = Subset(train_dataset, indices)
        subset = MyDataset(subset,transform_train)
        trainloader = DataLoader(subset, batch_size=16, shuffle=True)

        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        for param in model.parameters():
            param.requires_grad = False


        # Unfreeze some layers

        #for param in model.layer1.parameters():
        #   param.requires_grad = True

        #for param in model.layer2.parameters():
        #    param.requires_grad = True

        #for param in model.layer3.parameters():
        #    param.requires_grad = True
    
        for param in model.layer4.parameters():
            param.requires_grad = True
    


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
    
        # Convert model to be used on GPU
        model = model.to(device)



     
        list_models.append(model)
        train_dataloaders.append(trainloader) #subsammple of training data with replacement


    training(list_models, epochs, train_dataloaders, train_dataloader, val_dataloader, samples_size, len_train_dataset, len_val_dataset)
   

