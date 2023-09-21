#-------------------------Get statistics of the characteristics of the data in DICOM format needed for our experiments------------
#The data needed for our experiments to the data we selected from the big dataset and the data that are cleaned
#The characteristics are saved in a txt file


import numpy as np
import matplotlib.pyplot as plt

list_dates = []
list_sizes = []
list_labels = []
list_image_type = []
list_number_rows = []
list_number_columns = []

i=0
with open('data_dicom_cleaned_files.txt', 'r') as file:
    for row in file:
        if i==0:
            i+=1
            continue
       
        row = row.split()
               
        

        list_dates.append(row[0][:-1])
        list_sizes.append(int(row[1][:-1]))
        list_number_rows.append(row[4][:-1])
        list_number_columns.append(row[5][:-1])
        list_labels.append(row[6])
        
        i+=1


print("Maximum size ", np.max(list_sizes))
print("Minimum size ", np.min(list_sizes))
print("Mean size ", np.mean(list_sizes))
print("Median size ", np.median(list_sizes))



list_sizes = np.array(list_sizes)
print("number of data less than 5 MB ",list_sizes[list_sizes<5000000].shape) 



index_data_less_5 = np.where(list_sizes[list_sizes < 5000000])[0]
list_labels = np.array(list_labels)
list_labels_data_less_5 = list_labels[index_data_less_5]

labels_less_5 = np.unique(list_labels_data_less_5, return_counts=True)

print("Distribution class for images less than 5 MB ", labels_less_5)


print("\n")


list_tuple_columns_rows = list(zip(list_number_rows, list_number_columns))
print("All possible number of rows and columns ",  set(list_tuple_columns_rows))



index_4000_4000 = [index for index, item in enumerate(list_tuple_columns_rows) if item == ('4000', '4000')]
print("Number of images of shape (4000,4000) => ",len(index_4000_4000))

list_labels_4000_4000 = list_labels[index_4000_4000]
print("Distribution class for images of shape (4000,4000) => ", np.unique(list_labels_4000_4000, return_counts=True))

index_3072_3072 = [index for index, item in enumerate(list_tuple_columns_rows) if item == ('3072', '3072')]
print("Number of images of shape (3072,3072) => ",len(index_3072_3072))

list_labels_3072_3072 = list_labels[index_3072_3072]
print("Distribution class for images of shape (3072,3072) => ", np.unique(list_labels_3072_3072, return_counts=True))

index_2048_2600 = [index for index, item in enumerate(list_tuple_columns_rows) if item == ('2048', '2600')]
print("Number of images of shape (2048,2600) => ",len(index_2048_2600))

list_labels_2048_2600 = list_labels[index_2048_2600]
print("Distribution class for images of shape (2048,2600) => ", np.unique(list_labels_2048_2600, return_counts=True))

index_3072_3900 = [index for index, item in enumerate(list_tuple_columns_rows) if item == ('3072','3900')]
print("Number of images of shape (3072,3900) => ",len(index_3072_3900))

list_labels_3072_3900 = list_labels[index_3072_3900]
print("Distribution class for images (3072,3900) => ", np.unique(list_labels_3072_3900, return_counts=True))


