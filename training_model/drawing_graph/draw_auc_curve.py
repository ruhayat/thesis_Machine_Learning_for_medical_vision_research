#-------------------------------------Draw a plot of AUC values for any experiment--------------


import numpy as np
import matplotlib.pyplot as plt
import sys

onh_or_periphery = sys.argv[1]

liste_auc = []

with open("../metrics_test/auc_"+onh_or_periphery+"/auc.txt", "r") as file:

    for row in file:
        liste_auc.append(float(row.split()[2]))



plt.figure()

plt.plot(np.array([0,10,20,30,40,50,60]),liste_auc)

plt.xlabel('crops', fontsize = 14)
plt.ylabel('AUC', fontsize = 14)
plt.yticks([0.6,0.62,0.64,0.66,0.68,0.7])

if onh_or_periphery == "onh":
    plt.title('AUC values for the different crops from the optic nerve head', fontsize = 16)
else:
    plt.title('AUC values for the different crops from the periphery', fontsize = 16)

plt.legend()

plt.savefig("../graphs_metrics/auc_curve_"+onh_or_periphery+".png")


