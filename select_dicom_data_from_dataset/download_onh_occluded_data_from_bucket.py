#-------------------------------------Download the data that have been occluded from the onh-----------------------
#The data to download are in PNG format




from google.cloud import storage
import csv
import os
import pandas as pd
import numpy as np

client = storage.Client()
source_bucket_name = "hr-glaucoma-crops-dir"
#The source buckets are both 'hr-glaucoma-crops-dir' and 'hr-glaucoma-crops-dir-next'
bucket = client.get_bucket(source_bucket_name)
blobs = list(bucket.list_blobs())




i=1

for blob_ in blobs:
    
    if 'onh-00' in blob_.name :
        blob = bucket.blob(blob_.name)
        blob.download_to_filename("../data/onh/data_00_onh/"+blob_.name)
        print(i)
        i+=1
    

    elif 'onh-10' in blob_.name :
        blob = bucket.blob(blob_.name)
        blob.download_to_filename("../data/onh/data_10_onh/"+blob_.name)
        print(i)
        i+=1
    
    
    elif 'onh-20' in blob_.name:
        blob = bucket.blob(blob_.name)
        blob.download_to_filename("data/data-onh/data_20_onh/"+blob_.name) 
        print(i)
        i+=1
    
    elif 'onh-30' in blob_.name:
        blob = bucket.blob(blob_.name)
        blob.download_to_filename("data/onh/data_30_onh/"+blob_.name) 
        print(i)
        i+=1
        
    elif 'onh-40' in blob_.name:
        blob = bucket.blob(blob_.name)
        blob.download_to_filename("data/onh/data_40_onh/"+blob_.name) 
        print(i)
        i+=1
        
    elif 'onh-50' in blob_.name:
        blob = bucket.blob(blob_.name)
        blob.download_to_filename("data/onh/data_50_onh/"+blob_.name) 
        print(i)
        i+=1
        
    elif 'onh-60' in blob_.name:
        blob = bucket.blob(blob_.name)
        blob.download_to_filename("data/onh/data_60_onh/"+blob_.name) 
        print(i)
        i+=1
        

