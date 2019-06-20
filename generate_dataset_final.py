import RUCML.File.Dataset as Dataset

import RUCML.Utils.TerminalLogger as Log
import RUCML.Utils.Timer as Timer

import matplotlib.pyplot as plt
import numpy as np
import sys
import time

IMG_WIDTH = 64

filename = "dataset_pickle/dataset_" + str(IMG_WIDTH) + "_RAW_TRAINING_" + str(time.strftime("%Y%m%d")) + ".pickle"
Dataset.Generate("dataset_final", filename, IMG_WIDTH, "Training", grayscale = False)

filename = "dataset_pickle/dataset_" + str(IMG_WIDTH) + "_RAW_TESTING_" + str(time.strftime("%Y%m%d")) + ".pickle"
Dataset.Generate("dataset_final", filename, IMG_WIDTH, "Testing", grayscale = False)