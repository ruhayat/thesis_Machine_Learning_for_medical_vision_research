#-----------------------Get the characteristics of data as images in DICOM format needed for our experiments--------------
#The characteristics are the size (in bytes), the number of rows, the number of columns, the type of data... Those characteristics are saved in a txt file 
#The data needed for our experiments are those we selected from the original big dataset and those who are cleaned


import pydicom
import os
from google.cloud import storage
import pandas as pd



client = storage.Client()
bucket_name = 'exported-datasets-88282cb3-ripf-1918-alzeye-20230627-files'
bucket = client.bucket(bucket_name)


def download_data(dicom_file):
    # Replace 'source_blob_name' with the name of the file you want to download
    source_blob_name = "Images/" + dicom_file

    # Replace 'destination_file_name' with the name of the file you want to save locally
    destination_file_name = './dicom/'+dicom_file

    # Get the blob (object) you want to download
    blob = bucket.blob(source_blob_name)
    
    # Download the blob's contents to a local file
    blob.download_to_filename(destination_file_name)

    print("Download")




def get_image_type(dicom_file_path):

    if os.path.exists(dicom_file_path) == False:
        return False
    
    dicom_dataset = pydicom.dcmread(dicom_file_path)

    image_type = dicom_dataset.get("ImageType", "Type not available")

    return image_type



def get_number_rows_columns(dicom_file_path):

    if os.path.exists(dicom_file_path) == False:
        return False

    dicom_data = pydicom.dcmread(dicom_file_path)

    # Get the number of rows and columns
    num_rows = dicom_data.Rows
    num_columns = dicom_data.Columns

    return num_rows, num_columns



def get_date_and_size(dicom_file_path):


    # Check if the file exists
    if os.path.exists(dicom_file_path) == False:
        return False

    # Read the DICOM file
    dicom_dataset = pydicom.dcmread(dicom_file_path)

    # Extract the Acquisition Date if available
    acquisition_date = dicom_dataset.get("AcquisitionDateTime", "Date not available")

    # Get the file size in bytes
    file_size = os.path.getsize(dicom_file_path)


    return acquisition_date, file_size



def convert_png_into_dicom(png_file):

    dicom_file = png_file[:-15] + ".dcm"
    return dicom_file





if __name__ == "__main__":

    df = pd.read_csv("../wanted_data.csv")




    with open('./data_dicom_cleaned_files.txt','a') as f:
        f.write("date, size, image_type, num_rows, num_columns, label \n")


    list_files = os.listdir("../data/onh/data_00_onh/Images/data/")
    #we get the name of the PNG files which represent data we selected from the original big dataset and data after cleaning
    

    for f_png in list_files:
        
        #download the data
        f_dcm = convert_png_into_dicom(f_png)
        download_data(f_dcm)
        
        #get the characteristics from each dicom data
        date,size = get_date_and_size("./dicom/"+f_dcm)
        image_type = get_image_type("./dicom/"+f_dcm)
        num_rows, num_columns = get_number_rows_columns("./dicom/"+f_dcm)
        label = df[df["path"] == "Images/"+f_dcm]["labels"].iloc[0]
        
        #copy the characteristics in the txt file
        with open('./data_dicom_cleaned_files.txt', 'a') as f:
            f.write(str(date)+", "+str(size)+", "+str(image_type)+", "+str(num_rows)+", "+str(num_columns)+", "+str(label))
            f.write("\n")
        
        #remove the data after getting the characteristics to save some space
        os.system("rm ./dicom/"+f_dcm)

        

