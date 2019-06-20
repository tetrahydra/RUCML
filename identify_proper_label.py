import RUCML.File.Dataset as Dataset
from RUCML.File.Image import *
import RUCML.File.Utils as FileUtils
import RUCML.NeuralNetwork.ANN as ANN

import RUCML.Utils.GenUtils as GenUtils
import RUCML.Utils.TerminalLogger as Log
import RUCML.Utils.Timer as Timer
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.stats as stats
import sys
import time

' = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = '

'''
Identify
'''

IMG_WIDTH = 128

alter_dataset = ["NORM1", "NORM2", "STD"]
alter = 0

path_dataset = "/Users/eddie/RUC/_PROGRAMMING (MAIN RUCML)/dataset_final/"

' = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = '

image_to_identify1 = "/Users/eddie/RUC/YEAR 2 SEM 2/COMP SCIENCE SEMESTER PROJECT/2019-06-20 FINAL PRESENTATION/DELAUNAY/1_DSC_0037.JPG"
image_to_identify2 = "/Users/eddie/RUC/YEAR 2 SEM 2/COMP SCIENCE SEMESTER PROJECT/2019-06-20 FINAL PRESENTATION/DELAUNAY/output_composite.JPG"

' = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = '

if alter_dataset[alter] == "NORM1":
	trained_file = "/Users/eddie/RUC/YEAR 2 SEM 2/COMP SCIENCE SEMESTER PROJECT/A --- REPORT/CHAPTER 6 EXPERIMENTS & RESULTS/RESULTS FROM HPC/2019-06-05 NORM1 EPOCH/trained_model_3C_128_512_2000_NORM1_20190605.pickle"

if alter_dataset[alter] == "NORM2":
	trained_file = "/Users/eddie/RUC/YEAR 2 SEM 2/COMP SCIENCE SEMESTER PROJECT/A --- REPORT/CHAPTER 6 EXPERIMENTS & RESULTS/RESULTS FROM HPC/2019-06-05 NORM2/trained_model_3C_128_512_20190605.pickle"

if alter_dataset[alter] == "STD":
	trained_file = "/Users/eddie/RUC/YEAR 2 SEM 2/COMP SCIENCE SEMESTER PROJECT/A --- REPORT/CHAPTER 6 EXPERIMENTS & RESULTS/RESULTS FROM HPC/2019-06-05 STD/trained_model_3C_128_512_20190605_STD.pickle"
	
' = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = '

model_trained = FileUtils.PickleLoad(trained_file)

neural_network = model_trained[0]["network_architecture"]
model_parameters = model_trained[0]["model_parameters"]
volunteer_id = model_trained[0]["volunteer_id"]

' = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = '

print ("")

image_vector = ImageLoad(image_to_identify1, IMG_WIDTH, alter_dataset[alter])

prediction, Y_hat = ANN.Identify(neural_network, model_parameters, image_vector)

print ("Predicted Volunteer : " + str(volunteer_id[int(prediction.item())]))
print ("Confidence Level : " + ("{0:.2f} %".format(np.max(Y_hat) * 100)) )
#print ("Mean : " + str(np.abs(Y_hat - np.mean(Y_hat))))
print ("Standard Deviation : " + str(np.std(Y_hat)))
print ("Variance : " + str(np.var(Y_hat, ddof=1)))

' = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = '

print ("")

image_vector = ImageLoad(image_to_identify2, IMG_WIDTH, alter_dataset[alter])

prediction2, Y_hat2 = ANN.Identify(neural_network, model_parameters, image_vector)

print ("Predicted Volunteer : " + str(volunteer_id[int(prediction2.item())]))
print ("Confidence Level : " + ("{0:.2f} %".format(np.max(Y_hat2) * 100)) )
#print ("Mean : " + str(np.abs(Y_hat2-np.mean(Y_hat2))))
print ("Standard Deviation : " + str(np.std(Y_hat2)))
print ("Variance : " + str(np.var(Y_hat2, ddof=1)))

' = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = '

img_1 = plt.imread(image_to_identify1)

path_vol_id = path_dataset + str(volunteer_id[int(prediction.item())]) + "/ID-FaceForward"
for file in sorted(os.listdir(path_vol_id)):
	file_jpeg_name = os.path.join(path_vol_id, file)
	if file_jpeg_name.endswith(".jpg") or file_jpeg_name.endswith(".JPG"):
		photo_id = os.path.join(path_vol_id, file)
		
img_2 = plt.imread(photo_id)


font = {'family' : 'monospace',
        'weight' : 'bold',
        'size'   : 8}

plt.rc('font', **font)
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(11.69,8.27))

ax1 = axs[0,0]
ax2 = axs[0,1]
ax3 = axs[0,2]
ax4 = axs[1,0]
ax5 = axs[1,1]
ax6 = axs[1,2]

' = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = '

ax1.imshow(img_1)
ax1.axis('off')
ax1.set_title("Predicted as Volunteer " + str(volunteer_id[int(prediction2.item())]) + "\n Confidence: " + ("{0:.2f} %".format(np.max(Y_hat) * 100)) )
ax1.text(0.5,-0.05, "Photo without Label ID", size=8, ha="center", transform=ax1.transAxes)

ax2.imshow(img_2)
ax2.axis('off')
ax2.set_title("Volunteer " + str(volunteer_id[int(prediction.item())]) + " ↓ " )
ax2.text(0.5,-0.05, "Predicted Volunteer", size=8, ha="center", transform=ax2.transAxes)

ax3.set_title("SD: " + ("{0:.4f}".format(np.std(Y_hat))) + ", Var: "+ ("{0:.4f}".format(np.var(Y_hat, ddof=1))) )
ax3.plot(Y_hat * 100, marker=".")
ax3.set_ylabel('Confidence Level Distribution %')
ax3.set_xlabel('Volunteer ID')
x_tick = range(0, len(volunteer_id))
ax3.set_xticks(x_tick)
ax3.set_xticklabels(volunteer_id, rotation='vertical')

img_3 = plt.imread(image_to_identify2)

' = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = '

path_vol_id = path_dataset + str(volunteer_id[int(prediction2.item())]) + "/ID-FaceForward"
for file in sorted(os.listdir(path_vol_id)):
	file_jpeg_name = os.path.join(path_vol_id, file)
	if file_jpeg_name.endswith(".jpg") or file_jpeg_name.endswith(".JPG"):
		photo_id = os.path.join(path_vol_id, file)
		
img_4 = plt.imread(photo_id)

ax4.imshow(img_3)
ax4.axis('off')
ax4.set_title("Predicted as Volunteer " + str(volunteer_id[int(prediction2.item())]) + "\n Confidence: " + ("{0:.2f} %".format(np.max(Y_hat2) * 100)) )
ax4.text(0.5,-0.05, "Photo without Label ID", size=8, ha="center", transform=ax4.transAxes)

ax5.imshow(img_4)
ax5.axis('off')
ax5.set_title("Volunteer " + str(volunteer_id[int(prediction2.item())]) + " ↓ " )
ax5.text(0.5,-0.05, "Predicted Volunteer", size=8, ha="center", transform=ax5.transAxes)

ax6.set_title("SD: " + ("{0:.4f}".format(np.std(Y_hat2))) + ", Var: "+ ("{0:.4f}".format(np.var(Y_hat2, ddof=1))) )
ax6.plot(Y_hat2 * 100, marker=".")
ax6.set_ylabel('Confidence Level Distribution %')
ax6.set_xlabel('Volunteer ID')
x_tick = range(0, len(volunteer_id))
ax6.set_xticks(x_tick)
ax6.set_xticklabels(volunteer_id, rotation='vertical')

' = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = '


fig.tight_layout()
fig.savefig(str(time.strftime("%Y-%m-%d")) + " Plot Identify.png",bbox_inches='tight')
plt.show()

' = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = '


