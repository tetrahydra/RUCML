import numpy as np
import os
import pickle
from collections import defaultdict
from PIL import Image

import RUCML.File.Image as FileImage


def Generate(dataset_path, dataset_file, folder_name, IMG_WIDTH, image_format=None):
    dataset_photo = []
    dataset_label = []
    dataset_label_original = []

    image_file_types = ('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG')

    dataset_volunteer = defaultdict(lambda: defaultdict(int))

    volunteer_ids = []
    for volunteer_id in os.listdir(dataset_path):
        volunteer_path = os.path.join(dataset_path, volunteer_id)
        if os.path.isdir(volunteer_path):
            volunteer_ids.append(int(volunteer_id))

    volunteer_ids = sorted(volunteer_ids)
    volunteer_ids = list(map(str, volunteer_ids))

    for volunteer_id in volunteer_ids:

        volunteer_path = os.path.join(dataset_path, volunteer_id)
        i = 0

        if os.path.isdir(volunteer_path):
            for file_listing in sorted(os.listdir(volunteer_path)):
                if file_listing == folder_name:
                    sub_folder = os.path.join(volunteer_path, file_listing)

                    files = [file_i
                             for file_i in os.listdir(sub_folder)
                             if file_i.endswith(image_file_types)]

                    filenames = [os.path.join(sub_folder, fname)
                                 for fname in files]

                    for image_file in filenames:
                        dataset_volunteer[str(volunteer_id)][str(i)] = image_file
                        i += 1

    dataset_label_serial = 0
    for volunteer_id in volunteer_ids:
        for i in dataset_volunteer[volunteer_id]:
            if os.path.isfile(dataset_volunteer[volunteer_id][i]):
                if image_format == "Grayscale":
                    image = Image.open(dataset_volunteer[volunteer_id][i])
                    image = image.convert(mode='L')
                    image = image.resize((IMG_WIDTH, IMG_WIDTH), Image.ANTIALIAS)
                    image = np.array(image).reshape((IMG_WIDTH * IMG_WIDTH, 1))
                elif image_format == "LBP":
                    image = LBPH.pattern(dataset_volunteer[volunteer_id][i], IMG_WIDTH)
                elif image_format == "LBPH":
                    image = LBPH.histogram(dataset_volunteer[volunteer_id][i], IMG_WIDTH, 16)
                else:
                    image = Image.open(dataset_volunteer[volunteer_id][i])
                    image = image.resize((IMG_WIDTH, IMG_WIDTH), Image.ANTIALIAS)
                    image = np.array(image).reshape((IMG_WIDTH * IMG_WIDTH * 3, 1))

                vector_image = image.flatten()

                if len(dataset_photo) == 0:
                    dataset_photo = np.array(vector_image)
                    dataset_label = np.array(int(dataset_label_serial))

                else:
                    dataset_photo = np.vstack([dataset_photo, vector_image])
                    dataset_label = np.hstack([dataset_label, int(dataset_label_serial)])

            dataset_label_original.append(volunteer_id)

        dataset_label_serial += 1

    dataset = dataset_label
    dataset = np.vstack([dataset, dataset_label_original])
    dataset = np.vstack([dataset, dataset_photo.T])

    with open(dataset_file, "wb") as file:
        pickle.dump(dataset, file, protocol=pickle.HIGHEST_PROTOCOL)


def Load(filename, scaling=None):
    with open(filename, "rb") as f:
        dataset = pickle.load(f)

    dataset = dataset[:, np.random.permutation(dataset.shape[1])]

    dataset = dataset.astype(int)

    Y = dataset[0]

    ## Correct label for each feature
    ## set(C) = list of unique label
    ## len(set(C)) = total of unique label
    C = dataset[1]

    ## Remove Y
    dataset = np.delete(dataset, (0), axis=0)

    ## Remove C
    X = np.delete(dataset, (0), axis=0)

    if scaling != None:
        X = FileImage.ImageScaling(X, scaling)

    return X, Y, C


def one_hot_label(classification_total, tempY, m):
    tempY = np.reshape(tempY, (-1, m))
    Y = np.zeros((classification_total, m))
    for i in range(0, m):
        Y[tempY[0, i], i] = 1
    return Y
