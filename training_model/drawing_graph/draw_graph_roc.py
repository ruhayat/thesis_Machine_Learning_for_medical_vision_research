#--------------------------------------Drawing ROC curve for any experiment----------------


import numpy as np
import matplotlib.pyplot as plt
import sys

onh_or_periphery = sys.argv[1]

liste_FPR_00 = []
liste_TPR_00 = []

liste_FPR_10 = []
liste_TPR_10 = []

liste_FPR_20 = []
liste_TPR_20 = []

liste_FPR_30 = []
liste_TPR_30 = []

liste_FPR_40 = []
liste_TPR_40 = []

liste_FPR_50 = []
liste_TPR_50 = []

liste_FPR_60 = []
liste_TPR_60 = []


i=0
with open("../metrics_test/roc_"+onh_or_periphery+"/roc_metrics.txt", "r") as f:
    
    for row in f :
        liste = row.split()
        liste = liste[2:]

        for j in range(len(liste)):
            liste[j] = float(liste[j])

        if i==0:
            liste_FPR_00 = liste
        elif i==1:
            liste_TPR_00 = liste

        elif i==2:
            liste_FPR_10 = liste
        elif i==3:
            liste_TPR_10 = liste

        elif i==4:
            liste_FPR_20 = liste
        elif i==5:
            liste_TPR_20 = liste

        elif i==6:
            liste_FPR_30 = liste
        elif i==7:
            liste_TPR_30 = liste

        elif i==8:
            liste_FPR_40 = liste
        elif i==9:
            liste_TPR_40 = liste

        elif i==10:
            liste_FPR_50 = liste
        elif i==11:
            liste_TPR_50 = liste
    
        elif i==12:
            liste_FPR_60 = liste
        elif i==13:
            liste_TPR_60 = liste

        i+=1


plt.figure()

plt.plot(liste_FPR_00, liste_TPR_00, label='0%', color='blue')
plt.plot(liste_FPR_10, liste_TPR_10, label='10%', color='green')
plt.plot(liste_FPR_20, liste_TPR_20, label='20%', color='red')
plt.plot(liste_FPR_30, liste_TPR_30, label='30%', color='yellow')
plt.plot(liste_FPR_40, liste_TPR_40, label='40%', color='purple')
plt.plot(liste_FPR_50, liste_TPR_50, label='50%', color='orange')
plt.plot(liste_FPR_60, liste_TPR_60, label='60%', color='black')

plt.xlabel('FPR', fontsize = 14)
plt.ylabel('TPR', fontsize = 14)

if onh_or_periphery == "onh":
    plt.title('ROC Curve for the different crops from the optic nerve head', fontsize = 16)
else:
    plt.title('ROC Curve for the different crops from the periphery', fontsize = 16)

plt.legend()

plt.savefig("../graphs_metrics/roc_curve_"+onh_or_periphery+".png")






