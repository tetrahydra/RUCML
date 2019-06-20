import matplotlib.pyplot as plt
import numpy as np
import pickle
from PIL import Image


def ImageLoad(filename, img_width):
    image = Image.open(filename)
    image = image.resize((img_width, img_width), Image.ANTIALIAS)
    image = np.array(image).reshape((img_width * img_width * 3, 1))

    vector_image = image.flatten()

    dataset_image = np.array(vector_image)

    dataset_image = dataset_image / 255.
    dataset_image = dataset_image.reshape(dataset_image.shape[0], -1)

    return dataset_image


def PickleSave(filename, data_to_save):
    with open(filename, "wb") as file:
        pickle.dump(data_to_save, file, protocol=pickle.HIGHEST_PROTOCOL)


def PickleLoad(filename):
    file_to_open = open(filename, 'rb')
    readfile = pickle.load(file_to_open)
    return readfile


def PlotDraw(file_name, loss_history, printText = None):
    plt.figure(figsize=(11.69, 8.27))
    plt.rcParams["font.family"] = "Courier New"
    plt.gca().spines['right'].set_color('none')
    plt.gca().spines['top'].set_color('none')
    plt.plot(loss_history)
    plt.ylabel('Loss')
    plt.xlabel('Iteration')
    
    if printText is not None:
    	plt.text(0.02, 0.0, printText, transform=plt.gcf().transFigure)
    
    plt.savefig(file_name, bbox_inches='tight')
