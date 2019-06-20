import numpy as np
from PIL import Image


# Grayscale formula
# https://pillow.readthedocs.io/en/3.2.x/reference/Image.html#PIL.Image.Image.convert

def RGB2toGrayscale(RGB):
    return np.dot(RGB[..., :3], [0.2989, 0.5870, 0.1140])


def ImageLoad(filename, img_width, scaling = "NORM1"):
    image = Image.open(filename)
    image = image.resize((img_width, img_width), Image.ANTIALIAS)
    image = np.array(image).reshape((img_width * img_width * 3, 1))

    image_vector = image.flatten()

    image_vector = np.array(image_vector)

    return ImageScaling(image_vector, scaling)


def ImageVector(image, img_width, scaling = "NORM1", grayscale=False):
    if grayscale == True:
        image = image.convert(mode='L')
        image = image.resize((img_width, img_width), Image.ANTIALIAS)
        image = np.array(image)
        image = Image.fromarray(image)
        image = np.array(image).reshape((img_width * img_width, 1))
    else:
        image = image.resize((img_width, img_width), Image.ANTIALIAS)
        image = np.array(image).reshape((img_width * img_width * 3, 1))

    image_vector = image.flatten()
    image_vector = image_vector.astype(float)

    image_vector = np.array(image_vector)

    return ImageScaling(image_vector, scaling)


def ImageScaling(image_vector, scaling):
    image_vector = image_vector.astype(float)

    if scaling == "NORM1":
        image_vector = image_vector / 255.

    if scaling == "NORM2":
        image_vector = (image_vector / 255.) - 0.5

    if scaling == "STD":
        image_vector -= int(np.mean(image_vector))
        image_vector /= int(np.std(image_vector))

    image_vector = image_vector.reshape(image_vector.shape[0], -1)
    return image_vector
