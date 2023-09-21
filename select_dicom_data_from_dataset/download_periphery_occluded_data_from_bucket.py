#------------------------------------------Download the data that have been occluded from the periphery--------------------
#The data to download are in PNG format


from google.cloud import storage

client = storage.Client()

bucket = client.get_bucket("hr-glaucoma-crops-dir")
#The source buckets are both 'hr-glaucoma-crops-dir' and 'hr-glaucoma-crops-dir-next'

blobs = list(bucket.list_blobs())

i=1
for blob_ in blobs:
    
    if "periphery-00" in blob_.name: 

        blob = bucket.blob(blob_.name)
        blob.download_to_filename("data/periphery/periphery_00/"+blob_.name)
        print(i)
        i+=1

    elif "periphery-10" in blob_.name: 
        blob = bucket.blob(blob_.name)
        blob.download_to_filename("data/periphery/periphery_10/"+blob_.name)
        print(i)
        i+=1
    
    elif "periphery-20" in blob_.name: 
        blob = bucket.blob(blob_.name)
        blob.download_to_filename("data/periphery/periphery_20/"+blob_.name)
        print(i)
        i+=1

    elif "periphery-30" in blob_.name: 
        blob = bucket.blob(blob_.name)
        blob.download_to_filename("data/periphery/periphery_30/"+blob_.name)
        print(i)
        i+=1

    elif "periphery-40" in blob_.name: 
        blob = bucket.blob(blob_.name)
        blob.download_to_filename("data/periphery/periphery_40/"+blob_.name)
        print(i)
        i+=1

    elif "periphery-50" in blob_.name: 
        blob = bucket.blob(blob_.name)
        blob.download_to_filename("data/periphery/periphery_50/"+blob_.name)
        print(i)
        i+=1

    elif "periphery-60" in blob_.name: 
        blob = bucket.blob(blob_.name)
        blob.download_to_filename("data/periphery/periphery_60/"+blob_.name)
        print(i)
        i+=1

