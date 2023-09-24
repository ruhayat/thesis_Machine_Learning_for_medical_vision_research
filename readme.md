
This repository contains multiple folders. 

### select_dicom_data_from_dataset

#### select_dicom_data.py

A dataset has been provided from Moorfileds Eye Hospital. The dataset is stored in a google cloud platform account and the images are in dicom format.
In this dataset, in addition of the images, csv files have been provided as well. Two of them have been downloaded in my local machine. The first one is a table with each row representing a patient and containing the id of the patient and two different ways to label glaucoma (hesglaucoma glaucoma and mehgl). The second one is a table with each row representing one images and having the modality of that image, the id of the patient and the path to the image in dataset. 
The ids of patients can do the mapping between the images and the labels. 
For privacy reasons, the csv files are not shown in the repository.


In this project, the data we are in interested in have the modality "Optomap" and the binary label we will consider is 'hesglaucoma'.
It turns out that 3280 Optomap images are label 1 among 39958 Optomap images. To have a balanced dataset, we select 3280 Optomap images label 0 that are selected randomly and 3280 Optomap images label 1.

This script creates a csv file named 'wanted_data.csv' which is table containing only the images we have selected. Each row contains the id of the patient, the path to the image in the dataset and the label associated. 
Again for privacy reason, the created csv file is not shown in the repository. 


#### copy_selected_dicoms_to_new_bucket.py

 This script created a new bucket in the dataset containing only the data in dicom format we are interested in. In order to do so, the script copy those data from their original bucket to a new bucket.

#### preprocessing_code

The code has been provided by an external company and I am not allowed to show it on the repository. 
This code converts first the DICOM data into a PNG format. 
Then, extract the optic disc coordinates from the PNG data and, afterwards, crop the PNG data into 400 rows and 400 columns centered on the optic disc coordinates. 
Finally, the code occludes the [400,400] PNG images by applying a blacked out region of different sizes at the center. The occlusion occurs from both the onh and periphery, with for each, the different sizes corresponding to 0%,10,20%,30%,40%,50%,60% of the height of the images (400 pixels). 

It results in 14 sets of PNG data that are saved all together in a new bucket in the Google Cloud Platform account.

#### download_onh_occluded_data_from_bucket.py

The code downloads the [400,400] PNG images that have been occluded from the optic nerve head to the the local machine. It corresponds to 7 sets of data. Those data are not shown in the github repository for privacy reasons. 

#### download_periphery_occluded_data_from_bucket.py

The code downloads the [400,400] PNG images that have been occluded from the periphery to the the local machine. It corresponds to 7 sets of data. Those data are not shown in the github repository for privacy reasons. 







### data_cleaning

Cleaning the data consists of removing the outliers or removing the bad-quality data.
In this folder, two different methods are proposed to clean the data. Those two methods are applied on the 0% onh occluded PNG data. 
In this project, the first method has finally been chosen. 

#### cleaning_first_method.py

This method consists of removing manually the data that are susceptible of being bad-quality or outliers by setting up thresholds on the mean of the red channel and on the mean of the green channel.

#### cleaning_second_method.py

This method consists of removing automatically the data by using an OutofDistribution instance from cleanlab. I have selected 100 good-quality data and  15 bad-quality data. The instance is fitted on those data, and is afterward applied on the other data as explained in the report. 








### data_splitting

#### splitting_data_train_val_test.py

This script must be executed after cleaning the 0% onh occluded data.

To execute this file, two parameters must be provided. This first one is either 'onh' or 'periphery'. The second one is either '00','10','20','30','40','50' or '60'.

For instance, 'python splitting_data_train_val_test.py onh 00'


If the two parameters are 'onh' and '00', it will split the 0% onh occluded data into a training set, a validation set and a test set such a way that there is no patient having one image in the training set and one image in the validation or test set. 

If the two parameters are not 'onh' and '00', then it will apply exaclty the same split as for the 0% onh occluded data. 


Therefore, for any of the 14 sets, the training, validation and test set will be composed of exactly the same data. 
The split for the 0% onh occluded data must be done before splitting any other set. 

#### put_data_back.py

This script must be executed after the corresponding set of data has been splitted.

To execute this file, two parameters must be provided. This first one is either 'onh' or 'periphery'. The second one is either '00','10','20','30','40','50' or '60'.

For instance, 'python put_data_back.py onh 00'


This script will put all the data back together in the same folder after the data has been splitted. This script is just a way to backtrack right before splitting the data. 







### training_model

#### training.py

To execute this file, two parameters must be provided. This first one is either 'onh' or 'periphery'. The second one is either '00','10','20','30','40','50' or '60'.

For instance, 'python training.py onh 00'

This script trains a ResNet-50 model on the corresponding set of data. It includes the loss cross-entropy function, the Adam optimizer, the data augmentation techniques as detailed in the report. The metrics obtained during the train phase and during the validation phase are saved as txt files. 
At each epoch, the current model is saved. After training, the model having the best performances on validation data is chosen and the others are deleted. 

This script also includes commented lines about the simple CNN we used, the L2-regularistaion technique as detailed in the report. 


#### training_bagging.py

This file is only executed on the 0% occluded data. 
It implements a bagging method by training 5 ResNet-50 classifiers. 


#### test_model.py

To execute this file, two parameters must be provided. This first one is either 'onh' or 'periphery'. The second one is either '00','10','20','30','40','50' or '60'.

For instance, 'python training.py onh 00'

The script will get the right model and will apply the model on test data by calculating the AUC value, a list of true positive rates and a list of false negative rates (to compute the roc curve). 

The AUC value is saved in a file and the true positive rates and false negative rates are saved in another file. 


#### drawing_graph

##### draw_graph_metrics.py

To execute this file, two parameters must be provided. This first one is either 'onh' or 'periphery'. The second one is either '00','10','20','30','40','50' or '60'.

For instance, 'python draw_graph_metrics.py onh 00'

This script draws graphs of metrics throughout the epochs during the training phase and during the validation phase for the corresponding set of data. Moreover, the graph draws the loss function throughout the epochs for the training phase and for the validation phase. 

##### draw_graph_roc.py

To execute this file, one parameter must be provided. It is either 'onh' or 'periphery'. 
For instance, 'python draw_graph_roc.py onh'

This script draws a graph of ROC curves for either the onh set or the periphery set.

##### draw_auc_curve.py

To execute this file, one parameter must be provided. It is either 'onh' or 'periphery'. 
For instance, 'python draw_auc_curve.py onh'

This script draws a graph of AUC values for either the onh set or the periphery set. 






### data_volumetry

In this folder, the scripts calculate and plot statistics from the data. Some statistics are for example the number of patients, the maximum number of images per patient...

#### data_volumetry_before_cleaning.py

This script calculate and plot statistics from the data we selected before any cleaning is applied. 

#### data_volumetry_after_cleaning_before_splitting.py

This script calculate and plot statistics from the data we selected after a cleaning is applied and before spliting the data.

#### data_volumetry_after_cleaning_after_splitting.py

This script calculate and plot statistics from the data we selected separately on the training set, the validation set and the test set after a cleaning method is applied and after having splitted the data.







### dicom_data_parameters

#### get_characteristics_dicom_data.py

This script gets characteristics about the data, that have been selected from the original dataset and that have been cleaned, in their dicom format. 
It creates a txt file which is table, each row represents a dicom image containing the size, the number of rows, the number of columns...


#### get_statistics.py

This scripts reads the txt file containing the characteritics of the dicom images and get some statistics such as the maximum size, the minimum size, how many data with 4000 rows and 4000 columns. 






 
### plots_of_data

#### data_plots.py

This script only regards the 0% occluded data and plots some graphs. 

It plots the mean of the green channel and the mean of the red channel for each image in the scatter plot such a way that each image is a data point with 2 dimensions (mean of green channel, mean of red channel). 

It plots as well 2 bars graphs. One showing the mean of the red channel for each image and the other one showing the mean of the green chanel for each image. 



