import numpy as np

import RUCML.Maths.Activation as MathsActivation
import RUCML.Maths.Loss as MathsLoss

import RUCML.Utils.GenUtils as GenUtils

import scipy.stats as stats

printLog = GenUtils.PrintLog("LOG/hpc_run_log_ANN.txt")

def Initialization(NN):
    cache_parameters = {}
    epsilon = 0.25

    for index, layer in enumerate(NN):
        layer_index = index + 1
        layer_input_size, layer_output_size = layer["dimension_in"], layer["dimension_out"]

        cache_parameters['W' + str(layer_index)] = np.random.normal(loc=0.0,
                                                                    scale=np.sqrt(6) / np.sqrt(
                                                                        layer_input_size + layer_output_size),
                                                                    size=(layer_input_size, layer_output_size))

        cache_parameters['b' + str(layer_index)] = np.ones((layer_output_size, 1))

    return cache_parameters


def Training(NN, X, Y_expected, Y_hot, configuration, files, cache_parameters=None):
    learning_rate = configuration["learning_rate"]
    regularization_constant = configuration["regularization_constant"]
    mu = configuration["mu"]
    epochs = configuration["epochs"]

    loss = np.zeros(epochs)

    cache_momentum = {}
    for index, layer in enumerate(NN):
        layer_index = index + 1
        cache_momentum["v" + str(layer_index)] = 0
        cache_momentum["b_v" + str(layer_index)] = 0

    if cache_parameters == None:
        cache_parameters = Initialization(NN)
    else:
        print ("Using Cache")

    for i in range(0, epochs):
        ## Forward Propagation
        cache_calculations = Forward_Propagation(NN, cache_parameters, X)

        ## Calculate Loss
        loss[i] = MathsLoss.Cross_Entropy_Multiclass(cache_calculations["A" + str(len(NN))], Y_hot)

        ## Back Propagation
        cache_gradients = Back_Propagation(X, Y_hot, NN,
                                           cache_parameters,
                                           cache_calculations,
                                           regularization_constant)

        ## optimization : Momentum Gradient Descent
        cache_parameters, cache_momentum = Optimization(NN, cache_parameters, cache_gradients, cache_momentum,
                                                            learning_rate, mu)

        if (i + 1) % (epochs * 0.1) == 0:
            percent = float(i + 1) / epochs * 100
            printLog.text(str(percent) + "% Completed, Loss: " + str("{:.4f}".format(loss[i])))

    A_output = cache_calculations["A" + str(len(NN))]

    prediction = np.empty((1, X.shape[1]))
    for i in range(0, X.shape[1]):
        prediction[0, i] = np.argmax(A_output[:, i])

    mat1 = np.matrix(prediction)
    mat2 = np.matrix(Y_expected)
    with open(files["text-prediction"], 'ab') as f:
        for line in mat1:
            np.savetxt(f, line, fmt='%.0f')

        for line in mat2:
            np.savetxt(f, line, fmt='%.0f')

    accuracy = np.mean((prediction == Y_expected) * np.ones(prediction.shape)) * 100

    return cache_parameters, loss, accuracy


def Forward_Propagation(NN, cache_parameters, X):
    cache_calculations = {}

    cache_calculations["A0"] = X

    for i in range(1, int(len(NN) + 1)):
        cache_calculations["Z" + str(i)] = np.dot(cache_parameters["W" + str(i)].T,
                                                  cache_calculations["A" + str(i - 1)]) + cache_parameters["b" + str(i)]
        cache_calculations["A" + str(i)] = MathsActivation.Activate(cache_calculations["Z" + str(i)],
                                                                    NN[i - 1]["activation"])

    return cache_calculations


def Back_Propagation(X, y, NN, cache_parameters, cache_calculations, regularization_constant):
    (n, m) = X.shape

    delta = {}
    dL_dW = {}
    cache_gradients = {}

    # Problem with log(0) https://stackoverflow.com/questions/38125319/python-divide-by-zero-encountered-in-log-logistic-regression
    # Logistic Regression loss function
    epsilon = 1e-12
    cache_calculations["A" + str(len(NN))] = np.clip(cache_calculations["A" + str(len(NN))], epsilon, 1. - epsilon)

    ## delta = dL/dA * dA/dS : Simplified to A-Y
    delta[str(len(NN))] = MathsLoss.Softmax_Cross_Entropy_Derivative(cache_calculations["A" + str(len(NN))], y)

    for i in range(len(NN) - 1, 0, -1):
        ## delta = dL/dA * dA/dS
        delta[str(i)] = (cache_parameters["W" + str(i + 1)].dot(delta[str(i + 1)])) * MathsActivation.Derivative(
            cache_calculations["Z" + str(i)], NN[i - 1]["activation"])

    for i in range(len(NN), 0, -1):
        dL_dW[str(i)] = delta[str(i)].dot(cache_calculations["A" + str(i - 1)].T)
        cache_gradients["dW" + str(i)] = dL_dW[str(i)].T / m + (regularization_constant / m) * cache_parameters[
            "W" + str(i)]
        ## need to check the axis, is it correct?
        cache_gradients["db" + str(i)] = np.sum(dL_dW[str(i)], axis=1, keepdims=True) / m + (
                regularization_constant / m) * cache_parameters["b" + str(i)]

    return cache_gradients


def Optimization(NN, cache_parameters, cache_gradients, cache_momentum, learning_rate, mu):
    for index, layer in enumerate(NN):
        i = index + 1
        cache_momentum["v" + str(i)] = mu * cache_momentum["v" + str(i)] - learning_rate * cache_gradients[
            "dW" + str(i)]
        cache_parameters["W" + str(i)] += cache_momentum["v" + str(i)]

        cache_momentum["b_v" + str(i)] = mu * cache_momentum["b_v" + str(i)] - learning_rate * cache_gradients[
            "db" + str(i)]
        cache_parameters["b" + str(i)] += cache_momentum["b_v" + str(i)]

    return cache_parameters, cache_momentum


def Predict(NN, cache_parameters, X):
    cache_calculations = Forward_Propagation(NN, cache_parameters, X)
    A_predict = cache_calculations["A" + str(len(NN))]
    del cache_calculations
    return A_predict


def Identify(NN, cache_parameters, X):
    A_predict = Predict(NN, cache_parameters, X)
    Y_hat = A_predict
    fit = stats.norm.pdf(A_predict, np.mean(A_predict), np.std(A_predict))
    print ("Fit")
    print (fit)
    print ("Predicted Output")
    print (A_predict)
    print ("Sum Output")
    print (A_predict.sum(axis=0))
    A_predict[A_predict < 0.4] = -0.1
    print ("Altered Output")
    print (A_predict)
    prediction = np.empty((1, X.shape[1]))

    prediction[0, 0] = -1
    if len([*filter(lambda x: x >= 0, A_predict)]) > 0:
        for i in range(0, X.shape[1]):
            prediction[0, i] = np.argmax(A_predict[:, i])

    return prediction, Y_hat
