#------------------ Analyze the volumetry of the data before any cleaning and any splitting ------------------------
 


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset,Dataset, DataLoader
import torch
import torchvision.transforms as transforms
import os


df = pd.read_csv("../wanted_data.csv")
#the csv file wanted_data is a table of three columns :  the id of the patients, the path of the image in the GCP bucket, and the labels 




#Number of images

number_total_images = len(df["path"])
print("Number of all images ", number_total_images)

#Number of label 0 images

number_total_images_label_0 = len(df[df["labels"] == 0.0]["path"])
print("Number of all label 0 images ", number_total_images_label_0)

#Number of label 1 images

number_total_images_label_1 = len(df[df["labels"] == 1.0]["path"])
print("Number of all label 1 images ", number_total_images_label_1)


#Number of patients

number_total_patients = len(df["patients_ids"].unique())
print("Number of all patients ", number_total_patients)





count_number_of_images_per_patient = df.groupby("patients_ids").count()["path"].values


#Maximum number of images per patient 

maximum_number_of_images_per_patient = max(count_number_of_images_per_patient)
print("Maximum number of images per patient", maximum_number_of_images_per_patient)

#Minimum number of images per patient

minimum_number_of_images_per_patient = min(count_number_of_images_per_patient)
print("Minimum number of images per patient", minimum_number_of_images_per_patient)

#Mean number of images per patient

mean_number_of_images_per_patient = np.mean(count_number_of_images_per_patient)
print("Mean number of images per patient", mean_number_of_images_per_patient)

#Median number of images per patient

median_number_of_images_per_patient = np.median(count_number_of_images_per_patient)
print("Median number of images per patient", median_number_of_images_per_patient)



plt.figure()
plt.bar(np.arange(len(count_number_of_images_per_patient)),np.sort(count_number_of_images_per_patient))
plt.xlabel("number of patient")
plt.ylabel("number_of_data")
plt.title("Number of data per patient")
plt.savefig('graphs_volumetry_data/before_cleaning_and_splitting/number_of_data_per_patient.png')







