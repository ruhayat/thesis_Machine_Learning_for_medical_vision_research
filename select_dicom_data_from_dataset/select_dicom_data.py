#--------------------------------- From the Dicom images from the dataset, it selects the data we are interested in------------------------------
#The path in the bucket of the selected data, the patient id and the label for each of those data are saved in a csv file named 'wanted_data.csv'

#two csv files were downloaded from the original dataset : dcm_wide_080822.csv and images.csv
#dcm_wide_080822.csv contains the ids of patients and the hesglaucoma labels
#images.csv contains the ids of patients and the path of the images
#The mapping between the labels and the images is done through the patients' ids. Note that a patient id cannot have different label values.


import pandas as pd
import csv
import numpy as np
import random



df = pd.read_csv("./dcm_wide_080822.csv")

df_hesglaucoma = df[df["hesglaucoma"] == 1]
patientIds_hesglaucoma = df_hesglaucoma["dcm_id"].values



df_non_hesglaucoma = df[df["hesglaucoma"] == 0]

print("number of hesglaucoma ", df_hesglaucoma.shape)
print("number of non hesglaucoma ",df_non_hesglaucoma.shape)
patientIds_non_hesglaucoma = df_non_hesglaucoma["dcm_id"].values


patientId = []
hardwareModelName = []
modality = []
processedDicomGcsPath = []

with open("./images.csv", 'r') as file:
    csvreader = csv.reader(file)
    
    next(csvreader)
    for row in csvreader:
        patientId.append(row[0])
        hardwareModelName.append(row[7])
        modality.append(row[9])
        processedDicomGcsPath.append(row[-3])



patientId = np.array(patientId)
hardwareModelName = np.array(hardwareModelName)
modality = np.array(modality)
processedDicomGcsPath = np.array(processedDicomGcsPath)


indexes_optos_images = np.where((modality == "SLO - Red/Green") & (hardwareModelName == ''))[0]


indexes_patients_hesglaucoma = np.where(np.isin(patientId,patientIds_hesglaucoma))[0]


indexes_hesglaucoma_optos_images = np.intersect1d(indexes_optos_images, indexes_patients_hesglaucoma)


pathImages_hesglaucoma = processedDicomGcsPath[indexes_hesglaucoma_optos_images]


indexes_patients_non_hesglaucoma = np.where(np.isin(patientId, patientIds_non_hesglaucoma))[0]


indexes_non_hesglaucoma_optos_images = np.intersect1d(indexes_optos_images, indexes_patients_non_hesglaucoma)



nb_hesglaucoma_optos = indexes_hesglaucoma_optos_images.shape[0]
nb_non_hesglaucoma_optos = indexes_non_hesglaucoma_optos_images.shape[0]


indexes_non_hesglaucoma_optos_images = indexes_non_hesglaucoma_optos_images[random.sample(range(0,nb_non_hesglaucoma_optos),nb_hesglaucoma_optos)]


pathImages_non_hesglaucoma = processedDicomGcsPath[indexes_non_hesglaucoma_optos_images]


#Verification
i=0
for path in pathImages_non_hesglaucoma:
    index = np.where(processedDicomGcsPath == path)[0]

    #To make sure that the images are optomap images
    if modality[index] != "SLO - Red/Green" and hardwareModelName[index] != '':
        print("Error")

    #To make sure that the patient exists
    patient_id = patientId[index]
    if patient_id in df["dcm_id"].values and df[df["dcm_id"] == patient_id]["hesglaucoma"].values != 0:
        print("Error")
    
    print(i)
    i+=1




#creation wanted_data.csv file with one column for patient ids, one column for the path of the images and one column for the labels
pathImages = np.concatenate((pathImages_hesglaucoma, pathImages_non_hesglaucoma), axis=0)

labels = np.concatenate((np.ones((pathImages_hesglaucoma.shape[0],1)), np.zeros((pathImages_non_hesglaucoma.shape[0],1))), axis=0)

patients = np.concatenate((patientId[indexes_hesglaucoma_optos_images], patientId[indexes_non_hesglaucoma_optos_images]),axis=0)

data_wanted = np.column_stack((patients, pathImages, labels))

data_wanted = np.vstack((np.array(["patients_ids", "path","labels"]), data_wanted))

with open("../wanted_data.csv", "w", newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data_wanted)



