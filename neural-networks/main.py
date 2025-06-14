import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    
# loss function

def cross_entropy(predictions, targets):
    m = targets.shape[0]
    log_likelihood = -np.log(predictions[range(m), targets])
    loss = np.sum(log_likelihood) / m
    return loss