#-------------------------Put the training data, validation data and test data back all grouped together--------------

import os
import sys

onh_or_periphery = sys.argv[1] #either 'onh' or 'periphery'
cropping_size = sys.argv[2] #either '00','10','20','30','40','50' or '60'



if onh_or_periphery == "onh" and cropping_size == "00":

    #training data label 0
    os.system("mv ../data/onh/data_00_onh/training_images/Images/label_0/*.png ../data/onh/data_00_onh/Images/data/")
    
    #training data label 1
    os.system("mv ../data/onh/data_00_onh/training_images/Images/label_1/*.png ../data/onh/data_00_onh/Images/data/")

    #validation data label 0
    os.system("mv ../data/onh/data_00_onh/val_images/Images/label_0/*.png ../data/onh/data_00_onh/Images/data/")

    #validation data label 1
    os.system("mv ../data/onh/data_00_onh/val_images/Images/label_1/*.png ../data/onh/data_00_onh/Images/data/")

    #test data label 0
    os.system("mv ../data/onh/data_00_onh/test_images/Images/label_0/*.png ../data/onh/data_00_onh/Images/data/")
    
    #test data label 1
    os.system("mv ../data/onh/data_00_onh/test_images/Images/label_1/*.png ../data/onh/data_00_onh/Images/data/")

    
    os.system("mv ../data/onh/data_00_onh/Images/*.png ../data/onh/data_00_onh/Images/data/")


else:

    #training data label 0
    os.system("mv ../data/"+onh_or_periphery+"/data_"+cropping_size+"_"+onh_or_periphery+"/training_images/Images/label_0/*.png ../data/"+onh_or_periphery+"/data_"+cropping_size+"_"+onh_or_periphery+"/Images/")

    #training data label 1
    os.system("mv ../data/"+onh_or_periphery+"/data_"+cropping_size+"_"+onh_or_periphery+"/training_images/Images/label_1/*.png ../data/"+onh_or_periphery+"/data_"+cropping_size+"_"+onh_or_periphery+"/Images/")

    #validation data label 0
    os.system("mv ../data/"+onh_or_periphery+"/data_"+cropping_size+"_"+onh_or_periphery+"/val_images/Images/label_0/*.png ../data/"+onh_or_periphery+"/data_"+cropping_size+"_"+onh_or_periphery+"/Images/")

    #validation data label 1
    os.system("mv ../data/"+onh_or_periphery+"/data_"+cropping_size+"_"+onh_or_periphery+"/val_images/Images/label_1/*.png ../data/"+onh_or_periphery+"/data_"+cropping_size+"_"+onh_or_periphery+"/Images/")

    #test data label 0
    os.system("mv ../data/"+onh_or_periphery+"/data_"+cropping_size+"_"+onh_or_periphery+"/test_images/Images/label_0/*.png ../data/"+onh_or_periphery+"/data_"+cropping_size+"_"+onh_or_periphery+"/Images/")

    #test data label 1
    os.system("mv ../data/"+onh_or_periphery+"/data_"+cropping_size+"_"+onh_or_periphery+"/test_images/Images/label_1/*.png ../data/"+onh_or_periphery+"/data_"+cropping_size+"_"+onh_or_periphery+"/Images/")






