import numpy as np
import gc

def Geometric_Pyramid_Rule(layer_in, layer_out):

    return int(np.sqrt(layer_in + layer_out))

