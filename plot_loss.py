import RUCML.File.Dataset as Dataset
import RUCML.File.Utils as FileUtils
import RUCML.NeuralNetwork.ANN as ANN

import RUCML.Utils.TerminalLogger as Log
import RUCML.Utils.Timer as Timer

import matplotlib.pyplot as plt
import numpy as np
import sys
import time

loss = FileUtils.PickleLoad('2019-05-10/trained_loss_128_20190510_sigmoid_3L_geo.pickle')

FileUtils.PlotDraw("plot/plot_" + str(time.strftime("%Y%m%d_%H%M%S")) + ".png", loss)