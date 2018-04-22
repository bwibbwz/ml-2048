""" Neural network overlord

    Sets up input and neurons

    : no_in = number of inputs
    : no_nl = number of neuron layers
    : no_n  = number of neurons per layer
            int or array of ints

"""

from nn import *
from random import random

class overlord:

    def __init__(self, no_in, no_nl, no_n):

        self.no_in = no_in
        self.no_nl = no_nl
        self.no_n  = no_n

    def initialize(self):

        self.ol = []
        
        # Construct neural network
        l1 = InputLayer(self.no_in)
        self.ol.append(l1)

        for i in range(self.no_nl):
            n = NeuronLayer(self.no_n[i])
            self.ol.append(n)

    #def update_input(self):
    #    self.ol['input'] = random(self.nin)


ov = overlord(16,2,[4,16])
ov.initialize()

ov.ol
