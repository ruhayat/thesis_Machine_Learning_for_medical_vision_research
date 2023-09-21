#Script in order to draw a graph for any experiment representing the metrics : accuracy, recall and precision 


import numpy as np
import matplotlib.pyplot as plt
import sys



#Need to specify the experiment, the experiment can be defined by the cropping size and if it is either onh or periphery
cropping_size = sys.argv[2]
onh_or_periphery = sys.argv[1]


list_accuracy_training = []
list_precision_training = []
list_recall_training = []
list_loss_training = []

list_accuracy_val = []
list_precision_val = []
list_recall_val = []
list_loss_validation = []

i=0
with open("../metrics_validation/metrics_validation_"+cropping_size+"_"+onh_or_periphery+".txt", 'r') as file:
    
    for row in file:

        row = row.split() #convert into a list by splitting at each space
        accuracy = row[9][:-1]
        precision = row[12][:-1]
        recall = row[15][:-1]
        loss = row[6]
        
        #in the metrics.txt file, it alternate between row for training metrics and row for validation metrics
        if i%2==0:
            list_accuracy_training.append(float(accuracy))
            list_precision_training.append(float(precision))
            list_recall_training.append(float(recall))
            list_loss_training.append(float(loss))

        else:
            list_accuracy_val.append(float(accuracy))
            list_precision_val.append(float(precision))
            list_recall_val.append(float(recall))
            list_loss_validation.append(float(loss))
        
        i+=1
        


epochs = np.arange(0,40) + 1 #40 epochs


#----------- Plotting training metrics------------------------ 
plt.figure()

plt.plot(epochs, list_accuracy_training, label='accuracy', color='blue')
plt.plot(epochs, list_precision_training, label='precision', color='green')
plt.plot(epochs, list_recall_training, label='recall', color='red')


plt.xlabel('Epochs', fontsize = 14)
plt.ylabel('Metrics', fontsize = 14)

plt.title('Metrics by  epochs on training data', fontsize = 16)

plt.yticks([0,0.2,0.4,0.6,0.8,1])

plt.legend()

plt.savefig("../graphs_metrics/metrics_"+cropping_size+"_"+onh_or_periphery+"/metrics_training_data.png")



plt.figure()
plt.plot(epochs, list_loss_training, label='Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')




plt.title('Loss by epochs on training data')

plt.legend()

plt.savefig("../graphs_metrics/metrics_"+cropping_size+"_"+onh_or_periphery+"/loss_training_data.png")


#------------ Plotting validation metrics-----------------------
plt.figure()

plt.plot(epochs, list_accuracy_val, label='accuracy', color='blue')
plt.plot(epochs, list_precision_val, label='precision', color='green')
plt.plot(epochs, list_recall_val, label='recall', color='red')


plt.xlabel('Epochs',fontsize=14)
plt.ylabel('Metrics',fontsize=14)

plt.title('Metrics by  epochs on validation data',fontsize=16)
plt.yticks([0,0.2,0.4,0.6,0.8,1])

plt.legend()

plt.savefig("../graphs_metrics/metrics_"+cropping_size+"_"+onh_or_periphery+"/metrics_validation_data.png")



plt.figure()
plt.plot(epochs, list_loss_validation, label='Loss')
plt.xlabel('Epochs', fontsize=18)
plt.ylabel('Loss', fontsize=18)



plt.title("Loss by epochs on validation data", fontsize=18)

plt.legend()

plt.savefig("../graphs_metrics/metrics_"+cropping_size+"_"+onh_or_periphery+"/loss_validation_data.png")

