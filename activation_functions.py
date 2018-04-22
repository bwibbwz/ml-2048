from random import random
from math import exp, floor
import numpy as np

# ACTIVATION FUNCTIONS
class ActivationFunction():
    def __call__(self, input_value):
        if input_value is None:
            return None
        else:
            return self.transform(input_value)

class PassThrough(ActivationFunction):
    def transform(self, input_value):
        return input_value

class Scale(ActivationFunction):
    def __init__(self, scale_factor):
        self.scale_factor = scale_factor

    def transform(self, input_value):
        return self.scale_factor * input_value

class Sigmoid(ActivationFunction):
    def transform(self, input_value):
        return 1.0 / (1.0 + exp(-input_value))

class TanH(ActivationFunction):
    def transform(self, input_value):
        return (exp(input_value) - exp(-input_value)) / (exp(input_value) + exp(-input_value))

class ReLU(ActivationFunction):
    def transform(self, input_value):
        return max([0.0, input_value])

# This function has a wrong behaviour with input values of 0. This is done because the 2048 game won't care.
class Log2(ActivationFunction):
    def transform(self, input_value):
        if input_value == 0:
            return 0.0
        else:
            return np.log2(input_value)

class DiscreteAF(ActivationFunction):
    def __init__(self, number_of_steps, activation_function=PassThrough):
        self.number_of_steps = number_of_steps
        self.activation_function = activation_function()

    def transform(self, input_value):
        return int(floor(self.activation_function(input_value) * self.number_of_steps))

