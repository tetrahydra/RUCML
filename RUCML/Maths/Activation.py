import numpy as np
import gc


def Activate(sum, function):

	if function == "ReLU":
		return ReLU(sum)
	
	if function == "Sigmoid":
		return Sigmoid(sum)
	
	if function == "Softmax":
		return Softmax(sum)
	
	if function == "tanh":
		return tanh(sum)

def Derivative(activated, function):
	
	if function == "ReLU":
		return ReLU_Derivative(activated)
	
	if function == "Sigmoid":
		return Sigmoid_Derivative(activated)
	
	if function == "Softmax":
		return Softmax_Derivative(activated)
	
	if function == "tanh":
		return tanh_derivative(activated)

def ReLU(sum):
    activated = np.maximum(0, sum)
    gc.collect()
    return activated


def ReLU_Derivative(activated):
    derivation = np.array(activated, copy=True)
    derivation[activated <= 0] = 0;
    gc.collect()
    return derivation


def Sigmoid(sum):
    sum = np.clip(sum, -500, 500)

    activated = np.power(1.0 + np.exp(-sum), -1.0)
    gc.collect()
    return activated


def Sigmoid_Derivative(sum):
    derivation = Sigmoid(sum) * (1 - Sigmoid(sum))
    gc.collect()
    return derivation


def Softmax(sum):
    logC = np.max(sum)
    sum = sum + 1e-15

    if logC != 0:
        activated = np.exp(sum + logC) / np.sum(np.exp(sum + logC), axis=0, keepdims=True)
    else:
        activated = sum

    gc.collect()
    return activated


def Softmax_Derivative(Y_hat, Y):
    derivation = Y_hat - Y
    gc.collect()
    return derivation


def tanh(sum):
    activated = np.tanh(sum)
    gc.collect()
    return activated


def tanh_derivative(activated):
    derivation = 1.0 - np.tanh(activated) ** 2
    gc.collect()
    return derivation
