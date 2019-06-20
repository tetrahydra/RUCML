import RUCML.File.Dataset as Dataset
import RUCML.File.Utils as FileUtils
import RUCML.Maths.Maths as Maths
import RUCML.NeuralNetwork.ANN as ANN

import RUCML.Utils.GenUtils as GenUtils
import RUCML.Utils.TerminalLogger as Log
import RUCML.Utils.Timer as Timer

import matplotlib.pyplot as plt
import numpy as np
import sys
import time

Timer1 = Timer.Timer()

IMG_WIDTH = 128
EPOCH = 1000

hidden_1_size = 16

'''
External Files : START
'''

dataset1 = "../ANNML/dataset_pickle/Dataset_128_MIXED.pickle"

alter_dataset = ["NORM1", "NORM2", "STD"]
alter = 0

log_file = "LOG/HPC_Log_ANN_DEMO_" + str(IMG_WIDTH) + "_" + str(hidden_1_size) + ".txt"

predicted_file = "Prediction_DEMO_" + str(IMG_WIDTH) + "_" + str(hidden_1_size) + ".txt"

date = str(time.strftime("%Y%m%d"))

model_pickle = "trained_model_DEMO_"+str(IMG_WIDTH)+"_" + str(hidden_1_size) + "_" + date + "_"+str(alter_dataset[alter])+".pickle"
model_loss = "training_loss_DEMO_"+str(IMG_WIDTH)+"_" + str(hidden_1_size) + "_" + date + "_"+str(alter_dataset[alter])+".pickle"
model_accuracy_training = "accuracy_training_DEMO_"+str(IMG_WIDTH)+"_" + str(hidden_1_size) + "_" + date + "_"+str(alter_dataset[alter])+".pickle"
model_accuracy_testing = "accuracy_testing_DEMO_"+str(IMG_WIDTH)+"_" + str(hidden_1_size) + "_" + date + "_"+str(alter_dataset[alter])+".pickle"

'''
=    =    =    =    =    =    =    =    =    =    =    =    =    =    =    =
External Files : END
=    =    =    =    =    =    =    =    =    =    =    =    =    =    =    =
'''

printLog = GenUtils.PrintLog(log_file)

printLog.text("\n\nRunning at " + str(time.strftime("%Y-%m-%d %H:%M:%S")))

#     =    =    =    =    =    =    =    =    =    =    =    =    =    =    =    =     #

Timer1.Start()
printLog.text('Loading dataset for training...')
X, Y_expected, Y_original = Dataset.Load(dataset1, alter_dataset[alter])

printLog.text("Time taken to load dataset for training : " + str(Timer1.Value()))

# Count total classification
classification_total = len(set(Y_expected))

printLog.text('Preparing Label...')
Y_hot = Dataset.one_hot_label(classification_total, Y_expected, X.shape[1])

t = np.array(np.unique(Y_expected)).T
label_position = []

for i in range(0, len(t)):
    position_y_dataset = (np.where(Y_expected == i))[0][0]
    label_position.append(Y_original[position_y_dataset])

#     =    =    =    =    =    =    =    =    =    =    =    =    =    =    =    =     #

X_testing = []
Y_testing = []

#     =    =    =    =    =    =    =    =    =    =    =    =    =    =    =    =     #

print ("Hidden Size Layer : " + str(hidden_1_size))
print ("Dataset : " + str(dataset1))
print ("Dataset Total Sample : " + str(X.shape[1]))
print ("Dataset Shape : " + str(X.shape[0]) + "x" + str(X.shape[1]))
print ("Total Classification : " + str(classification_total))
print ("Total Epochs: " + str(EPOCH))

#     =    =    =    =    =    =    =    =    =    =    =    =    =    =    =    =     #

'''
Configurations
'''

neural_network = [
    {"dimension_in": X.shape[0], "dimension_out": hidden_1_size, "activation": "Sigmoid" },
    {"dimension_in": hidden_1_size, "dimension_out": classification_total, "activation": "Softmax"}
]

configuration = {
    "learning_rate": 0.075,
    "regularization_constant": 0.05,
    "mu": 0.5,
    "epochs": EPOCH,
}

files = {
    "text-prediction": predicted_file
}

#     =    =    =    =    =    =    =    =    =    =    =    =    =    =    =    =     #

'''
Training
'''

Timer1.Start()
printLog.text("Training...")

model_parameters, loss, accuracy_training, accuracy_testing = ANN.Training(neural_network,
                                                                           X,
                                                                           Y_expected,
                                                                           Y_hot,
                                                                           configuration,
                                                                           files,
                                                                           X_testing,
                                                                           Y_testing)

printLog.text("Time taken for training : " + str(Timer1.Value()))

#     =    =    =    =    =    =    =    =    =    =    =    =    =    =    =    =     #

'''
Save Information Into Pickle Files
'''

trained = [
    {
        "network_architecture": neural_network,
        "model_parameters": model_parameters,
        "volunteer_id": label_position
    }
]

FileUtils.PickleSave(model_pickle, trained)
#FileUtils.PickleSave(model_loss, loss)
#FileUtils.PickleSave(model_accuracy_training, accuracy_training)

#if len(accuracy_testing) > 0:
    #FileUtils.PickleSave(model_accuracy_testing, accuracy_testing)

#     =    =    =    =    =    =    =    =    =    =    =    =    =    =    =    =     #

printLog.text("Accuracy Training : " + str(accuracy_training[-1]))
if len(accuracy_testing) > 0:
	printLog.text("Accuracy Testing : " + str(accuracy_testing[-1]))

printLog.text("\nTraining finished at " + str(time.strftime("%Y-%m-%d %H:%M:%S")))

#     =    =    =    =    =    =    =    =    =    =    =    =    =    =    =    =     #
