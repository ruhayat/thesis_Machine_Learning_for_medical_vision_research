#---------------Copy the selected DICOM images from the original bucket to a new bucket


from google.cloud import storage
import csv


client = storage.Client()
source_bucket_name = "exported-datasets-88282cb3-ripf-1918-alzeye-20230627-files"
destination_bucket_name = "hr-glaucoma-dicoms"
#The destination buckets are boh 'hr-glaucoma-dicoms' and 'hr-glaucoma_dicoms-next'

source_bucket = client.get_bucket(source_bucket_name)
destination_bucket = client.get_bucket(destination_bucket_name)
 

i = 0   
list_files = []
with open("../wanted_data.csv", 'r') as file:
    csvreader = csv.reader(file)
    next(csvreader)
    for row in csvreader :
        
        if i % 10 == 0:
            print(i)

        dcmfile = row[1]
        dcmfile = dcmfile[7:]
        source_blob = source_bucket.blob("Images/"+dcmfile)
        source_bucket.copy_blob(source_blob, destination_bucket, dcmfile)
        i+=1
