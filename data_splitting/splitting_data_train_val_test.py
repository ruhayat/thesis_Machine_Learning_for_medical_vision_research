#----------------------------------------Split the data into a training set, a validation set and a test set----------------------------
#the split is created only  for the data at 0% occlusion from the onh, so for the data without applying any occlusion
#the same split is applied for the occluded data

import os
import pandas as pd
from torchvision.datasets import ImageFolder
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
import sys


cropping_size = sys.argv[2] #either '00','10','20','30','40','50' or '60'
onh_or_periphery = sys.argv[1]  #either 'onh' or 'periphery'



#Those three small functions are only used when the data to split are not the PNG images without any occlusion 

def change_filename_from_00_onh(filename_to_change):
    """
    We change the filename to its corresponding filename given its size of occlusion and whther it is occluded from onh or periphery
    """
    new_filename = filename_to_change.replace("00.png",cropping_size+".png",1)
    new_filename= new_filename.replace("onh", onh_or_periphery)

    return new_filename
    

def change_filenames_from_00_onh_in_list(list_filenames):
    """
    The list_filenames as an argument contains the PNG filenames for data without occlusion.
    In this function, we change each filename in the list to its corresponding filename given its size of occlusion and whether it is occluded from onh or periphery
    """
    new_list = []
    for filename in list_filenames:
        new_list.append(change_filename_from_00_onh(filename))
    
    return new_list


def move_files_from_00_onh(list_filenames, folder, label):
    """
    folder : "training_images", "val_images" or "test_images"
    label : "label_0" or "label_1"
    """

    for filename in list_filenames:
        os.system("mv ../data/"+onh_or_periphery+"/data_"+cropping_size+"_"+onh_or_periphery+"/Images/"+filename+" ../data/"+onh_or_periphery+"/data_"+cropping_size+"_"+onh_or_periphery+"/"+folder+"/Images/"+label+"/")








if cropping_size == "00" and onh_or_periphery == "onh":


    data_dir = "/home/rhayat_mehresearch_org/data/onh/data_00_onh/Images/data"
    filenames = os.listdir(data_dir)
    #list of cleaned data
    #the images are in the PNG format
    

    filename_labels = pd.read_csv("../wanted_data.csv")
    #dataframe containing the patients' ids, the path and the labels of all the data (good-quality and bad-quality)
    # the images in the path in the DICOM format

    liste_filename_retained = []





    for index,row in filename_labels.iterrows():
    
    
        filename = row["path"]
        modified_filename = filename[:-4]
        modified_filename = modified_filename[7:] + "-400-onh-00.png"

        if modified_filename in filenames:

            liste_filename_retained.append(row["path"])

   
    

    subset_interested_df = filename_labels[filename_labels["path"].isin(liste_filename_retained)]
    #dataframe containing the path of cleaned data only, the patients' ids and the labels for each data 



    #splitting into training set, validation and test set such that there is no one image in the train set and one image in any of the evaluation set (validation or test) 
    #belonging yo the same patient


    X = subset_interested_df["path"]
    y = subset_interested_df["labels"]
    groups = subset_interested_df["patients_ids"]



    sgkf = StratifiedGroupKFold(n_splits=5)

    for i, (train_index, test_index) in enumerate(sgkf.split(X, y, groups)):
        break

    
    #sgkf_2 = StratifiedGroupKFold(n_splits=8)
    X_train =  subset_interested_df["path"].iloc[train_index]
    y_train = subset_interested_df["labels"].iloc[train_index]
    groups_train = subset_interested_df["patients_ids"].iloc[train_index]



    sgkf_2 = StratifiedGroupKFold(n_splits=8)
    
    for i, (train_index_bis, val_index) in enumerate(sgkf_2.split(X_train,y_train,groups_train)):
        break

    val_index = train_index[val_index]
    train_index = train_index[train_index_bis]

    #we got the indexes for the training data, the validation data and the test data


    #training data
    liste_filenames_train = subset_interested_df["path"].iloc[train_index]
    for filename in liste_filenames_train:

        modified_filename = filename[:-4]
        modified_filename = modified_filename[7:] + "-400-"+onh_or_periphery+"-"+cropping_size+".png"
        #convert from the dcm path to the png filename

        label = filename_labels[filename_labels["path"]==filename]["labels"]
     
    
        if len(label) == 1 :
            if label.iloc[0] == 0:
                    #if label is 0, the png filename is moved to the label 0 training folder
                    os.system("mv ../data/"+onh_or_periphery+"/data_"+cropping_size+"_"+onh_or_periphery+"/Images/data/"+modified_filename+" ../data/"+onh_or_periphery+"/data_"+cropping_size+"_"+onh_or_periphery+"/training_images/Images/label_0/")
            
            elif label.iloc[0] == 1:
                    #if label is 1, the png filenames is moved to the label 1 training folder
                    os.system("mv ../data/"+onh_or_periphery+"/data_"+cropping_size+"_"+onh_or_periphery+"/Images/data/"+modified_filename+" ../data/"+onh_or_periphery+"/data_"+cropping_size+"_"+onh_or_periphery+"/training_images/Images/label_1/ ")




    #validation data
    liste_filenames_val = subset_interested_df["path"].iloc[val_index]
    for filename in liste_filenames_val:

        modified_filename = filename[:-4]
        modified_filename = modified_filename[7:] + "-400-"+onh_or_periphery+"-"+cropping_size+".png"
        #convert from the dcm path to the png filename
        


        label = filename_labels[filename_labels["path"]==filename]["labels"]
   
    
    
        if len(label) == 1 :
            if label.iloc[0] == 0:
                    #if label is 0, the png filename is moved to the label 0 validation folder 
                    os.system("mv ../data/"+onh_or_periphery+"/data_"+cropping_size+"_"+onh_or_periphery+"/Images/data/"+modified_filename+" ../data/"+onh_or_periphery+"/data_"+cropping_size+"_"+onh_or_periphery+"/val_images/Images/label_0/")
            elif label.iloc[0] == 1:
                    #if label is 1, the png filename is moved to the label 1 validation folder 
                    os.system("mv ../data/"+onh_or_periphery+"/data_"+cropping_size+"_"+onh_or_periphery+"/Images/data/"+modified_filename+" ../data/"+onh_or_periphery+"/data_"+cropping_size+"_"+onh_or_periphery+"/val_images/Images/label_1/ ")



    #test data
    liste_filenames_test = subset_interested_df["path"].iloc[test_index]
    for filename in liste_filenames_test:

        modified_filename = filename[:-4]
        modified_filename = modified_filename[7:] + "-400-"+onh_or_periphery+"-"+cropping_size+".png"
        #convert from the dcm path to the png filename
        

        label = filename_labels[filename_labels["path"]==filename]["labels"]
   
    
        #patient_id = filename_labels[filename_labels["path"]==filename]["patients_ids"].values[0]
    

        if len(label) == 1 :
            if label.iloc[0] == 0:
                    #if label is 0, the png filename is moved to the label 0 test folder 
                    os.system("mv ../data/"+onh_or_periphery+"/data_"+cropping_size+"_"+onh_or_periphery+"/Images/data/"+modified_filename+" ../data/"+onh_or_periphery+"/data_"+cropping_size+"_"+onh_or_periphery+"/test_images/Images/label_0/")
            elif label.iloc[0] == 1:
                    #if label is 1, the png filename is moved to the label 1 test folder
                    os.system("mv ../data/"+onh_or_periphery+"/data_"+cropping_size+"_"+onh_or_periphery+"/Images/data/"+modified_filename+" ../data/"+onh_or_periphery+"/data_"+cropping_size+"_"+onh_or_periphery+"/test_images/Images/label_1/ ")





else:
#we keep the same split for data that are  occluded


    training_files_00_onh_label_0 = os.listdir("../data/onh/data_00_onh/training_images/Images/label_0")
    training_files_00_onh_label_1 = os.listdir("../data/onh/data_00_onh/training_images/Images/label_1")
    validation_files_00_onh_label_0 = os.listdir("../data/onh/data_00_onh/val_images/Images/label_0")
    validation_files_00_onh_label_1 = os.listdir("../data/onh/data_00_onh/val_images/Images/label_1")
    test_files_00_onh_label_0 = os.listdir("../data/onh/data_00_onh/test_images/Images/label_0")
    test_files_00_onh_label_1 = os.listdir("../data/onh/data_00_onh/test_images/Images/label_1")

    
    new_training_files_00_onh_label_0 = change_filenames_from_00_onh_in_list(training_files_00_onh_label_0) 
    new_training_files_00_onh_label_1 = change_filenames_from_00_onh_in_list(training_files_00_onh_label_1) 
    new_validation_files_00_onh_label_0 = change_filenames_from_00_onh_in_list(validation_files_00_onh_label_0)
    new_validation_files_00_onh_label_1 = change_filenames_from_00_onh_in_list(validation_files_00_onh_label_1) 
    new_test_files_00_onh_label_0 = change_filenames_from_00_onh_in_list(test_files_00_onh_label_0) 
    new_test_files_00_onh_label_1 = change_filenames_from_00_onh_in_list(test_files_00_onh_label_1) 
    

    move_files_from_00_onh(new_training_files_00_onh_label_0, "training_images", "label_0")
    move_files_from_00_onh(new_training_files_00_onh_label_1, "training_images", "label_1")
    move_files_from_00_onh(new_validation_files_00_onh_label_0, "val_images", "label_0")
    move_files_from_00_onh(new_validation_files_00_onh_label_1, "val_images", "label_1")
    move_files_from_00_onh(new_test_files_00_onh_label_0, "test_images", "label_0")
    move_files_from_00_onh(new_test_files_00_onh_label_1, "test_images", "label_1")






     
