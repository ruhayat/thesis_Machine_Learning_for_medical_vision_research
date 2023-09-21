#--------------------------------Sceond cleaning method--------------------------





import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset,Dataset, DataLoader, ConcatDataset
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from PIL import Image
import os
from cleanlab.outlier import OutOfDistribution
from cleanlab.rank import find_top_issues
import timm
import torchvision
from subprocess import call


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


transform_tensor = transforms.Compose([transforms.ToTensor()])





def embed_images(model, dataloader):
    feature_embeddings = []
    for data in dataloader:
        images, labels = data
        images = images.to(device)
        with torch.no_grad():
            embeddings = model(images)
            feature_embeddings.extend(embeddings.cpu())

        i+=1
    feature_embeddings = np.array(feature_embeddings)
    return np.stack(feature_embeddings)  # each row corresponds to embedding of a different image


def fit_model_score():

    #we have selected 100 good-quality images and 15 bad-quality data 
    liste_filenames = ["HXAXOX3QZM5LPIMXUYMKVKHXGY-400-onh-00.png",
                    "DVVTSD22LCPS7KNQAZS6DKSX3E-400-onh-00.png",
                    "CC7F7YSQFGXJHHNFOD4KLTZIT4-400-onh-00.png",
                    "6PY36AS553Q5FIJDYXEPITE5Y4-400-onh-00.png",
                    "5AMRKVJN566XQW6SWSOVEIN72I-400-onh-00.png",
                    "RL72XFTQDEHYYCXWZ6LJXJ24XE-400-onh-00.png",
                    "3AWUFQFZBFDFB7SNLC4CAT4ZJ4-400-onh-00.png",
                    "KEEN4ZQR4QOIM6PFBNZXQRUL3Q-400-onh-00.png",
                    "L6M4BARRIYPIIULDH3MXU6CI3Q-400-onh-00.png",
                    "2PMCM24ZCV7TVDQCDV7PC2CP6Q-400-onh-00.png",
                    "FLCZVH4AZV3T4O3R4OVHHM533U-400-onh-00.png",
                    "DNUSPPGVSP5MPGNML742NQR2HQ-400-onh-00.png",
                    "SHRRVB7A75WTBMZ5PYTOGI5OBU-400-onh-00.png",
                    "DAI4VOSZKNS3RFWM5NO5BPDB5E-400-onh-00.png",
                    "64LQA3B4SAT5MKFQGFQJO4TL34-400-onh-00.png",
                    "6LZNZGOFDOIBLKUI6JKJVTARZE-400-onh-00.png",
                    "7CGTARSYWDKFPQDSDOLDM3IIKI-400-onh-00.png",
                    "7PE3R33CUL563ZWZK6GXQVMJGU-400-onh-00.png",
                    "AG5S2QSBYNEHFZWJL7XBXH5YME-400-onh-00.png",
                    "AMZGRKWEYUUU2UESROCQ7TY2CU-400-onh-00.png",
                    "AVGHYVQ6AR47YKGEFI42UGR6F4-400-onh-00.png",
                    "BBCSEDJFMVHI6ERASRMHANYQPY-400-onh-00.png",
                    "BQTC2HJMOA2FFEYJL553EXYIJM-400-onh-00.png",
                    "BUSO5A63WW4ZJ7KMMLFOQZBGYQ-400-onh-00.png",
                    "C25MPDD7WPKMIJKXERRBPC5VJU-400-onh-00.png",
                    "C2CW5QPH5DHMMU5BPIZQYA434U-400-onh-00.png",
                    "CCNLNOKARZ2IMKTJCT2HELODVQ-400-onh-00.png",
                    "CHEVD6XZNLGZ2FMR7OW3FHI5RM-400-onh-00.png",
                    "CHJZHDOKEULGZKJIQJ5I2OYUAU-400-onh-00.png",
                    "CI4W7WCEW5YRCXXQVGE5NDEJVE-400-onh-00.png",
                    "DBXJJG46K4ULHSAYB5IJCRSM5A-400-onh-00.png",
                    "DDHUUP4E7J5FEBFN7ULUB5LVBU-400-onh-00.png",
                    "DDPNFZQNWY4TXKV47VCKKVYVKQ-400-onh-00.png",
                    "DJOWNDNXR5LGXEBY6NPXZP4VYA-400-onh-00.png",
                    "DSZOPRFFRMVFOAHET4FK2JEVCY-400-onh-00.png",
                    "EAEWA7526JNRRIQ7JNVUJRQVIM-400-onh-00.png",
                    "EDX2VMVETL45PFJB4ISFKDKT5Y-400-onh-00.png",
                    "ETXWDEYFLFXY3E5AI2WHCE7DHA-400-onh-00.png",
                    "EZW7X7DM5DQTCBGQDIEMIRTGBM-400-onh-00.png",
                    "FCTMWYR3R4H3MNUTTDHGM6DUDM-400-onh-00.png",
                    "FMGREI2KA4EQBNFYSARFZPCPHM-400-onh-00.png",
                    "FON3X47E627FOND3AANEPSDKMY-400-onh-00.png",
                    "FT6KSIZZNJR2VGZTGGRLWB6DZM-400-onh-00.png",
                    "GPNGM4VMJIAXLNXGECAC7OSIQA-400-onh-00.png",
                    "GUQGJTOBTVPBKPEG4P47DBFV4M-400-onh-00.png",
                    "H7C5OBXMEWZIUT7LPCR367WA4A-400-onh-00.png",
                    "HDFML2GX7MSXH57PYVF3EQZGAU-400-onh-00.png",
                    "HPTPR4EWBBNSSBN4JZPGUMCZZU-400-onh-00.png",
                    "HZFGIHSIQDHS5IDG6G2RNJE4VY-400-onh-00.png",
                    "HZVWU5LUAAC4DKPRNEUERVU6GQ-400-onh-00.png",
                    "I33ZJ7WQBCAPL3RHZDE3OATAQM-400-onh-00.png",
                    "IGPHW7GHZDBOINVDAKMR65VUN4-400-onh-00.png",
                    "IP5Y7H7WBCBLPC47H3POAU75XA-400-onh-00.png",
                    "JFHF73PC3SFNTGVOUF6IXJ5ZWA-400-onh-00.png",
                    "G34GYITRTMRN3R7ZT53I6VRPPY-400-onh-00.png",
                    "EBWUU2AQ4RMA4LCQSSUAOD3VIQ-400-onh-00.png",
                    "DOJOGHJC5CWUOFY3LJK6KA4SLM-400-onh-00.png",
                    "C5GVQDTSFW6HWOVGF3ZZWABPBQ-400-onh-00.png",
                    "COR22B5U4GA775HRRTVDPZLVTE-400-onh-00.png",
                    "H3I62LAOXMKU4V6HEAEDFGSGRA-400-onh-00.png",
                    "GN234Y6U3QM5XXQE74FRRGIN4U-400-onh-00.png",
                    "N3NS5QJRUPWSC3AXYTI5ARVWZQ-400-onh-00.png",
                    "SYSLOAORGAP3E2X4XFHPLPM47U-400-onh-00.png",
                    "RSE27MVKTGPQDTEED2SRDFIFCA-400-onh-00.png",
                    "KHDITVHVK6JPMUQFCJZMCVY7AY-400-onh-00.png",
                    "7KBDLMCG4SGTH54Q2OKJPGHL3E-400-onh-00.png",
                    "4F4ZDAPFCFR3QYOEDVW42L7S6M-400-onh-00.png",
                    "77JK22W23LGVN4UU66X7VHC4E4-400-onh-00.png",
                    "SUAWILQTGNRACDINHFYGMPWDEM-400-onh-00.png",
                    "4LO2UI5WW5GCZ5BA26IBWEKZFY-400-onh-00.png",
                    "62SK4U5YMTFHKH3PTLPQ5IXY5Y-400-onh-00.png",
                    "6BHX4LEU3HGDBCV4LLCWWRQPJU-400-onh-00.png",
                    "6GTS3NUNAA33425FWM3GNGN7WQ-400-onh-00.png",
                    "6SUYB3DX3GFLAIK7YAJNUBA3IM-400-onh-00.png",
                    "7J5D6LPZ764UPXULJOIQQNIPRQ-400-onh-00.png",
                    "7LMMMB4GAJ3M4QA2HI3HGYPUNI-400-onh-00.png",
                    "7NF47I6OXX7BVU4VXYZESRXFGE-400-onh-00.png",
                    "7TVHGRUD2PRSZQJDJF2CVXD62Q-400-onh-00.png",
                    "7VHD3HBKI7RLTFJS43L5AZSBZI-400-onh-00.png",
                    "A3YDLNPEO6Y7W6Y5Z4R5IA7DVA-400-onh-00.png",
                    "A63ZOK6SWEKHQJM3SWYLYC7GX4-400-onh-00.png",
                    "B4TBGPYNXK2JF4TSTYT7NWN4F4-400-onh-00.png",
                    "CGV5QF3OGHYMK4ITVHMPK2IPGI-400-onh-00.png",
                    "CJNUOMQSFFII7CSWVJWLFWKGVQ-400-onh-00.png",
                    "CMBRWTIDJZKMLW6G7LDDV57J2M-400-onh-00.png",
                    "CU5V74V2MZKJZYLBPUOU4ZFSHI-400-onh-00.png",
                    "CU66OZ3QMCOOD5W6XIUCSMIJ3E-400-onh-00.png",
                    "FCYOAQABAUW65EQ4NFBNDLTKKU-400-onh-00.png",
                    "FWUYBZ6N6SHMCRZXG7F6XAXBRA-400-onh-00.png",
                    "G4Y2ANZ3ZRRM2MI7MGQAHIB7P4-400-onh-00.png",
                    "G72L7P35AGL33RVI6A7S5REIHM-400-onh-00.png",
                    "GRU7BJPRC5ZQ74FDBLG6MG6CIA-400-onh-00.png",
                    "GWV4X2MWY63J54XYB2CODRE2T4-400-onh-00.png",
                    "H2FALVAXD5H27K36T2LJA4WUGE-400-onh-00.png",
                    "HCVEXFJAHI7WQPDBG23UIW6QOA-400-onh-00.png",
                    "HKEKQLBIXZ42XNCF5MJ74JYOFE-400-onh-00.png",
                    "I2HVPHC3AZBRH6N32W43ZTDKUY-400-onh-00.png",
                    "I2KZ2WW3ZZSOL7YHGXJF37ISNE-400-onh-00.png",
                    "I4BV2M7TQXEOYNWO6E2UGT3MXE-400-onh-00.png",
                    "I75JWKZDV5A7F4PKT2SHPEFSGM-400-onh-00.png",
                    "IONEWKEGLHACBYUTWVKZBCAWNM-400-onh-00.png",
                    "GDDHHDKPAI5L2P47EEBPKUQVOA-400-onh-00.png",
                    "GGUWDACLS3SZWV7A5TY2SXUHMY-400-onh-00.png",
                    "GXX4S5DPL6UNVUS5ITTSSIKQFI-400-onh-00.png",
                    "IORTDVXN776WDEVOWBGN64QEVA-400-onh-00.png",
                    "IR5YN45DZGQTWBBBJRDHCUQGOE-400-onh-00.png",
                    "ITLF5LK266ZFABELI4LX2H6B5Q-400-onh-00.png",
                    "IU5TE63M5ED6J4Z6CMJ4NSFADQ-400-onh-00.png",
                    "JF6J6KNXI7XGH3MWORZO42A6VE-400-onh-00.png",
                    "IUDGE6DQCO5ZOMYHSQ5TCNFK7I-400-onh-00.png",
                    "7WLIFJEJRZZJ745B4ZBP4ATIY4-400-onh-00.png",
                    "HAPAZFIW73LPBHH27FUU72344U-400-onh-00.png",
                    "BNYE4R43CTSOWHQSUIIZYO4ECA-400-onh-00.png",
                    "72BY4OFFX6O5SZIZ5HPW7MZVAU-400-onh-00.png",
                    "7CWE2HY33NZ5GQCS3GTUVSGGGY-400-onh-00.png",
                    "DX6UVTVMUY7TFPE4EP7TPB5PWI-400-onh-00.png"
                    
                    ]
        
    if not os.path.exists("../data/onh/data_00_onh/test_good_data_bad_data"):
        os.system("mkdir ../data/onh/data_00_onh/test_good_data_bad_data")
        os.system("mkdir ../data/onh/data_00_onh/test_good_data_bad_data/Images")
        
    else:
        
        os.system("rm -r ../data/onh/data_00_onh/test_good_data_bad_data")
        os.system("mkdir ../data/onh/data_00_onh/test_good_data_bad_data")
        os.system("mkdir ../data/onh/data_00_onh/test_good_data_bad_data/Images")
        

    for filename in liste_filenames:
        os.system("mv ../data/onh/data_00_onh/Images/data/"+filename+" ../data/onh/data_00_onh/test_good_data_bad_data/Images")

    dataset = ImageFolder("../data/onh/data_00_onh/test_good_data_bad_data", transform = transform_tensor)
    dataloader = DataLoader(dataset, batch_size = 16, shuffle=True,num_workers=16)

    model = timm.create_model('resnet50', pretrained=True, num_classes=0)  # this is a pytorch network
    model.eval()  # eval mode disables training-time operators (like batch normalization)
    model.to(device)
        
    feature_embeddings = embed_images(model,dataloader)
    print(f'Embeddings pooled shape: {feature_embeddings.shape}')

    #train the instance on the 100 good-quality data and 15 bad-quality data
    ood = OutOfDistribution()
    ood_features_scores = ood.fit_score(features=feature_embeddings)
        
    for filename in liste_filenames:
        os.system("mv ../data/onh/data_00_onh/test_good_data_bad_data/Images/"+filename+" ../data/onh/data_00_onh/Images/data")
         
    os.system("rm -r ../data/onh/data_00_onh/test_good_data_bad_data")




    root_dir_data="../data/onh/data_00_onh/Images/"
    dataset = ImageFolder(root_dir_data, transform=transform_tensor)
    dataloader = DataLoader(dataset, batch_size=16,num_workers=16)



    feature_embeddings = embed_images(model,dataloader)
    print(f'Embeddings pooled shape: {feature_embeddings.shape}')

     
    
    ood_features_scores = ood.score(features=feature_embeddings)

    #third_percent score     
    third_percent = np.percentile(ood_features_scores,75)   
        
    #data having a score inferior thant the third percentile score are not retained
    non_outliers_idxs = np.where(ood_features_scores > third_percent)[0]
    
    liste_retained_filenames = []
    for i in non_outliers_idxs:
        liste_retained_filenames.append(dataset.imgs[i][0][-41:])

         
    return liste_retained_filenames



def remove_uncorrect_png_files(data_dir, liste_retained_filenames):
    

    filenames = os.listdir(data_dir)
    
    for filename in filenames:
        if filename not in liste_retained_filenames:
            os.system("mv "+data_dir+"/"+filename+ " /home/rhayat_mehresearch_org/data/onh/data_00_onh/Images/")
        
        i+=1




liste_retained_filenames  = fit_model_score()

remove_uncorrect_png_files("../data/onh/data_00_onh/Images/data",liste_retained_filenames)




