import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


lbp_uniform = {
    0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 6: 0, 7: 0, 8: 0, 12: 0, 14: 0,
    15: 0, 16: 0, 24: 0, 28: 0, 30: 0, 31: 0, 32: 0, 48: 0, 56: 0, 60: 0,
    62: 0, 63: 0, 64: 0, 96: 0, 112: 0, 120: 0, 124: 0, 126: 0, 127: 0, 128: 0,
    129: 0, 131: 0, 135: 0, 143: 0, 159: 0, 191: 0, 192: 0, 193: 0, 195: 0, 199: 0,
    207: 0, 223: 0, 224: 0, 225: 0, 227: 0, 231: 0, 239: 0, 240: 0, 241: 0, 243: 0,
    247: 0, 248: 0, 249: 0, 251: 0, 252: 0, 253: 0, 254: 0, 255: 0, 256: 0
}


uniform_bins = [0, 1, 2, 3, 4, 6, 7, 8, 12, 14, 15, 16, 24, 28, 30, 31, 32, 48, 56, 60, 62, 63, 64, 96, 112, 120, 124,
                126, 127, 128, 129, 131, 135, 143, 159, 191, 192, 193, 195, 199, 207, 223, 224, 225, 227, 231, 239, 240,
                241, 243, 247, 248, 249, 251, 252, 253, 254, 255]


def thresholding(image, center, x, y):
    threshold = 0
    try:
        if image[x][y] >= center:
            threshold = 1
    except:
        pass
    return threshold


def lbp_calculate(image, x, y):
    center = image[x][y]
    binary_value = []                                               # NEIGHBOURS
    binary_value.append(thresholding(image, center, x - 1, y + 1))  # top_right
    binary_value.append(thresholding(image, center, x, y + 1))      # right
    binary_value.append(thresholding(image, center, x + 1, y + 1))  # bottom_right
    binary_value.append(thresholding(image, center, x + 1, y))      # bottom
    binary_value.append(thresholding(image, center, x + 1, y - 1))  # bottom_left
    binary_value.append(thresholding(image, center, x, y - 1))      # left
    binary_value.append(thresholding(image, center, x - 1, y - 1))  # top_left
    binary_value.append(thresholding(image, center, x - 1, y))      # top

    binary_power = [1, 2, 4, 8, 16, 32, 64, 128]
    value_base10 = 0
    for i in range(len(binary_value)):
        value_base10 += binary_value[i] * binary_power[i]
    return int(value_base10)


def pattern(image, image_width):
    image = Image.open(image)
    image = image.convert(mode='L')
    image = image.resize((image_width, image_width), Image.ANTIALIAS)
    image = np.array(image)

    img_lbp = np.zeros((image_width, image_width), np.uint8)

    for i in range(0, image_width):
        for j in range(0, image_width):
            img_lbp[i, j] = lbp_calculate(image, i, j)

    return np.asarray(img_lbp).reshape((image_width * image_width, 1))


def histogram(image, image_width, region = 6):
    image = Image.open(image)
    image = image.convert(mode='L')
    image = image.resize((image_width, image_width), Image.ANTIALIAS)
    image = np.array(image)

    region_width = int(image_width / region)

    lbp_histogram_final = []

    for y in range (0, region):
        for x in range(0, region):
            region_sub = image[y * region_width:y * region_width + region_width, x * region_width:x * region_width + region_width]

            img_lbp = np.zeros((region_width, region_width), np.uint8)
            lbp_uniform_temp = lbp_uniform.copy()
            for i in range(0, region_width):
                for j in range(0, region_width):
                    img_lbp[i, j] = lbp_calculate(region_sub, i, j)
                    if img_lbp[i, j] in uniform_bins:
                        lbp_uniform_temp[img_lbp[i, j]] += 1
                    else:
                        lbp_uniform_temp[256] += 1

            lbp_histogram_final.extend(lbp_uniform_temp.values())
            del img_lbp, lbp_uniform_temp

    return np.asarray(lbp_histogram_final).reshape((len(lbp_histogram_final), 1))