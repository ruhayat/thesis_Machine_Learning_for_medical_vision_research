#----------------------- Analyze the volumetry of the data after cleaning and after splitting ---------------------------



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset,Dataset, DataLoader
import torch
import torchvision.transforms as transforms
import os



#training data after cleaning
list_cleaned_training_data_label_0 = os.listdir("../data/onh/data_00_onh/training_images/Images/label_0")
list_cleaned_training_data_label_1 = os.listdir("../data/onh/data_00_onh/training_images/Images/label_1")
list_cleaned_training_data = list_cleaned_training_data_label_0 + list_cleaned_training_data_label_1


#validation_data_after_cleaning
list_cleaned_validation_data_label_0 = os.listdir("../data/onh/data_00_onh/val_images/Images/label_0")
list_cleaned_validation_data_label_1 = os.listdir("../data/onh/data_00_onh/val_images/Images/label_1")
list_cleaned_validation_data = list_cleaned_validation_data_label_0 + list_cleaned_validation_data_label_1


#test_data_after_cleaning
list_cleaned_test_data_label_0 = os.listdir("../data/onh/data_00_onh/test_images/Images/label_0")
list_cleaned_test_data_label_1 = os.listdir("../data/onh/data_00_onh/test_images/Images/label_1")
list_cleaned_test_data = list_cleaned_test_data_label_0 + list_cleaned_test_data_label_1




df = pd.read_csv("../wanted_data.csv")



list_path_cleaned_training_data = []

for name in list_cleaned_training_data:
    name = name[:-15] + ".dcm"
    name = "Images/"+name
    list_path_cleaned_training_data.append(name)

df_training_data = df[df["path"].isin(list_path_cleaned_training_data)]



list_path_cleaned_validation_data = []

for name in list_cleaned_validation_data:
    name = name[:-15] + ".dcm"
    name = "Images/"+name
    list_path_cleaned_validation_data.append(name)

df_validation_data = df[df["path"].isin(list_path_cleaned_validation_data)]



list_path_cleaned_test_data = []

for name in list_cleaned_test_data:
    name = name[:-15] + ".dcm"
    name = "Images/"+name
    list_path_cleaned_test_data.append(name)

df_test_data = df[df["path"].isin(list_path_cleaned_test_data)]



#-----Training data--------

#Number of all training data
number_total_training_data = len(list_cleaned_training_data)
print("Number of all training data ", number_total_training_data)

#Number of all label 0 training data
number_total_training_data_label_0 = len(list_cleaned_training_data_label_0)
print("Number of all training data label 0", number_total_training_data_label_0)

#Number of all label 1 training data
number_total_training_data_label_1 = len(list_cleaned_training_data_label_1)
print("Number of all training data label 1", number_total_training_data_label_1)


count_number_training_images_per_patient = df_training_data.groupby("patients_ids").count()["path"].values

#Number of patients training
number_patients_training = len(df_training_data["patients_ids"].unique()) 
print("Number of patients in training data", number_patients_training)

#Maximum number of training data per patient
max_number_training_data_per_patient = max(count_number_training_images_per_patient)
print("Maximum number of training data per patient ", max_number_training_data_per_patient)

#Minimum number of training data per patient
min_number_training_data_per_patient = min(count_number_training_images_per_patient)
print("Minimum number of training data per patient ", min_number_training_data_per_patient)

#Mean number of training data per patient
mean_number_training_data_per_patient = np.mean(count_number_training_images_per_patient)
print("Mean number of training data per patient ", mean_number_training_data_per_patient)


#Median number of training data per patient
median_number_training_data_per_patient = np.median(count_number_training_images_per_patient)
print("Median number of training data per patient ", median_number_training_data_per_patient)



plt.figure()
plt.bar(np.arange(len(count_number_training_images_per_patient)),np.sort(count_number_training_images_per_patient))
plt.xlabel("number of patient")
plt.ylabel("number_of_data")
plt.title("Number of data per patient")
plt.savefig('graphs_volumetry_data/after_cleaning_and_after_splitting/number_of_training_data_per_patient.png')




#-----Validation data--------

#Number of all validation data
number_total_validation_data = len(list_cleaned_validation_data)
print("Number of all validation data ", number_total_validation_data)

#Number of all label 0 validation data
number_total_validation_data_label_0 = len(list_cleaned_validation_data_label_0)
print("Number of all validation data label 0", number_total_validation_data_label_0)

#Number of all label 1 validation data
number_total_validation_data_label_1 = len(list_cleaned_validation_data_label_1)
print("Number of all validation data label 1", number_total_validation_data_label_1)


count_number_validation_images_per_patient = df_validation_data.groupby("patients_ids").count()["path"].values

#Number of patients validation
number_patients_validation = len(df_validation_data["patients_ids"].unique()) 
print("Number of patients in validation data", number_patients_validation)


#Maximum number of validation data per patient
max_number_validation_data_per_patient = max(count_number_validation_images_per_patient)
print("Maximum number of validation data per patient ", max_number_validation_data_per_patient)

#Minimum number of validation data per patient
min_number_validation_data_per_patient = min(count_number_validation_images_per_patient)
print("Minimum number of validation data per patient ", min_number_validation_data_per_patient)

#Mean number of validation data per patient
mean_number_validation_data_per_patient = np.mean(count_number_validation_images_per_patient)
print("Mean number of validation data per patient ", mean_number_validation_data_per_patient)

#Median number of validation data per patient
median_number_validation_data_per_patient = np.median(count_number_validation_images_per_patient)
print("Median number of validation data per patient ", median_number_validation_data_per_patient)


plt.figure()
plt.bar(np.arange(len(count_number_validation_images_per_patient)),np.sort(count_number_validation_images_per_patient))
plt.xlabel("number of patient")
plt.ylabel("number_of_data")
plt.title("Number of data per patient")
plt.savefig('graphs_volumetry_data/after_cleaning_and_after_splitting/number_of_validation_data_per_patient.png')



#-----Test data--------

#Number of all test data
number_total_test_data = len(list_cleaned_test_data)
print("Number of all test data ", number_total_test_data)

#Number of all label 0 test data
number_total_test_data_label_0 = len(list_cleaned_test_data_label_0)
print("Number of all test data label 0", number_total_test_data_label_0)

#Number of all label 1 test data
number_total_test_data_label_1 = len(list_cleaned_test_data_label_1)
print("Number of all test data label 1", number_total_test_data_label_1)


count_number_test_images_per_patient = df_test_data.groupby("patients_ids").count()["path"].values

#Number of patients test
number_patients_test = len(df_test_data["patients_ids"].unique()) 
print("Number of patients in test data", number_patients_test)



#Maximum number of test data per patient
max_number_test_data_per_patient = max(count_number_test_images_per_patient)
print("Maximum number of test data per patient ", max_number_test_data_per_patient)

#Minimum number of test data per patient
min_number_test_data_per_patient = min(count_number_test_images_per_patient)
print("Minimum number of test data per patient ", min_number_test_data_per_patient)

#Mean number of test data per patient
mean_number_test_data_per_patient = np.mean(count_number_test_images_per_patient)
print("Mean number of test data per patient ", mean_number_test_data_per_patient)

#Median number of test data per patient
median_number_test_data_per_patient = np.median(count_number_test_images_per_patient)
print("Median number of test data per patient ", median_number_test_data_per_patient)


plt.figure()
plt.bar(np.arange(len(count_number_test_images_per_patient)),np.sort(count_number_test_images_per_patient))
plt.xlabel("number of patient")
plt.ylabel("number_of_data")
plt.title("Number of data per patient")
plt.savefig('graphs_volumetry_data/after_cleaning_and_after_splitting/number_of_test_data_per_patient.png')


