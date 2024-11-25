import random
import math
import sys
import time
from typing import List
RANDOM_WEIGHT_RANGE = 0.1
DEFAULT_LEARNING_RATE = 0.01
LOG_EPSILON = 1e-15

class Layer:
    def __init__(self):
        self.input = None
        self.preactivation = None
        self.output = None
        self.weight_deltas = None
        self.bias_deltas = None
        self.weights = None
        self.biases = None
    def activate(self, Z, A, activation: str, derivative: bool = False):
        activation = activation.lower()
        if activation == 'relu':
            self.reLU(Z, A, derivative=derivative)
        elif activation == 'softmax':
            if derivative:
                raise Exception("Derivative of softmax should not be computed directly.")
            else:
                self.softmax(Z, A)
        else:
            raise Exception(f"Unknown activation function '{activation}'")
    # assuming l has even row and column distibution
    @staticmethod
    def shape(_list):
        shape = []
        while True:
            if isinstance(_list, list):
                shape.append(len(_list))
                _list = _list[0]
            else:
                break
        return shape

