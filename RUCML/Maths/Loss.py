import numpy as np
import gc

import RUCML.Maths.Activation as MathsActivation

def MSE(Y_predicted, Y_expected):

    ## Convert to column matrix : dim[row, 1]
    Y_predicted = Y_predicted.reshape(-1,1)
    Y_expected = Y_expected.reshape(-1, 1)

    ## Sum by row and divide by total element in Y_expected
    loss = 1.0 / (2.0 * Y_expected.shape[0]) * np.sum(np.power(Y_predicted - Y_expected, 2), axis=0)

    gc.collect()

    ## Return a scalar value
    return loss.item()

def Cross_Entropy_Multiclass(Y_predicted, Y_expected):
    loss = - 1 / Y_expected.shape[1] * np.sum(Y_expected * np.log(Y_predicted))
    #probs = out / np.sum(Y_predicted, axis=1, keepdims=True) ## need to check dimension, is it axis 0 or 1
    #loss = np.sum( -np.log(probs[range(Y_expected.shape[1]), Y_expected]) )
    #loss += reg_lambda / 2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3)))
    #loss = 1./_expected.shape[1] * loss
    return loss

def Softmax_Cross_Entropy_Derivative(Y_predicted, Y_expected):
    grad = Y_predicted - Y_expected
    return grad

def Softmax_Derivative(Y_predicted, Y_expected):
    Y_predicted[range(Y_expected.shape[1]), Y_expected] -= 1
    return Y_predicted

