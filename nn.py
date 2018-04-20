from random import random
from math import exp, floor

# ACTIVATION FUNCTIONS
class ActivationFunction():
    def __call__(self, input_value):
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

class ReLU(ActivationFunction):
    def transform(self, input_value):
        return max([0.0, input_value])

class DiscreteSigmoid(ActivationFunction):
    def __init__(self, number_of_steps):
        self.number_of_steps = number_of_steps
        self.sigmoid = Sigmoid()

    def transform(self, input_value):
        return int(floor(self.sigmoid(input_value) * self.number_of_steps))

# OTHER FUNCTIONS
class RandomWrapper():
    def __call__(self):
        return random()

# ACTIVATION FUNCTION DEFAULTS
NEURON_ACTIVATION_FUNCTION = Sigmoid()
WEIGHT_ACTIVATION_FUNCTION = PassThrough()
NODE_ACTIVATION_FUNCTION = PassThrough()
INPUT_ACTIVATION_FUNCTION = PassThrough()
OUTPUT_ACTIVATION_FUNCTION = DiscreteSigmoid(4)

# OTHER DEFULATS
WEIGHT_INITIAL_VALUE = RandomWrapper()

# NODES, NEURONS AND WEIGHTS
class Node(object):
    def __init__(self, initial_value=None, activation_function=NODE_ACTIVATION_FUNCTION):
        self.set_activation_function(activation_function)
        self.set_value(initial_value)

    def set_activation_function(self, activation_function):
        self.activation_function = activation_function

    def get_value(self):
        return self.value

    def set_value(self, value):
        self.value = self.activation_function(value)

    def __repr__(self):
        return str(self.value)

class Neuron(Node):
    def __init__(self, previous_layer, initial_value=0.0, activation_function=NEURON_ACTIVATION_FUNCTION):
        super(Neuron, self).__init__(initial_value = initial_value, activation_function = activation_function)
        self.set_previous_layer(previous_layer)

        self.set_weights_layer(WeightLayer(self, self.get_previous_layer(), len(self.get_previous_layer())))

    def get_weights_layer(self):
        return self.weights

    def set_weights_layer(self, weights):
        self.weights = weights

    def get_previous_layer(self):
        return self.previous_layer

    def set_previous_layer(self, previous_layer):
        self.previous_layer = previous_layer

    def update_value(self):
        pl = self.get_previous_layer()
        wl = self.get_weights_layer()
        if len(pl) != len(wl):
            raise ValueError('The lengths of the previous layer (%i) does not mach the amount of weights(%i)' % (len(pl), len(wl)))
        new_value = 0.0
        for k in range(len(pl)):
            new_value += pl[k].get_value() * wl[k].get_value()
        self.set_value(new_value)

class Weight(Node):
    def __init__(self, node_in, node_out, initial_value=WEIGHT_INITIAL_VALUE):
        self.node_in = node_in
        self.node_out = node_out
        super(Weight, self).__init__(initial_value = initial_value, activation_function = WEIGHT_ACTIVATION_FUNCTION)

# LAYERS
class Layer(list):
    def __init__(self, number_of_items, item_class=Node, **kwargs):
        for k in range(number_of_items):
            super(Layer, self).append(item_class(**kwargs))
        self.set_previous_layer(None)
        self.set_weights_layer(None)

    def get_weights_layer(self):
        return self.weights

    def set_weights_layer(self, weights):
        self.weights = weights

    def get_previous_layer(self):
        return self.previous_layer

    def set_previous_layer(self, previous_layer):
        self.previous_layer = previous_layer

    def update_layer(self):
        for item in self:
            item.update_value()

class WeightLayer(Layer):
    def __init__(self, node, previous_layer, number_of_items, item_class=Weight, **kwargs):
        for k in range(number_of_items):
            super(Layer, self).append(item_class(previous_layer[k], node, initial_value = WEIGHT_INITIAL_VALUE()))
        self.set_previous_layer(previous_layer)
    
    def update_layer(self):
        pass
        
class NeuronLayer(Layer):
    def __init__(self, previous_layer, number_of_items, item_class=Neuron, **kwargs):
        super(NeuronLayer, self).__init__(number_of_items, item_class, previous_layer = previous_layer, **kwargs)
        self.set_previous_layer(previous_layer)
    
        weights = []
        for item in self:
            weights.append(item.get_weights_layer())

        self.set_weights_layer(weights)

class InputLayer(Layer):
    def __init__(self, number_of_items, item_class=Node, **kwargs):
        super(InputLayer, self).__init__(number_of_items, item_class, activation_function = INPUT_ACTIVATION_FUNCTION, **kwargs)
        self.set_previous_layer(None)

    def update_layer(self):
        pass
        
class OutputLayer(Layer):
    def __init__(self, previous_layer, number_of_items, item_class=Neuron, **kwargs):
        super(OutputLayer, self).__init__(number_of_items, item_class, previous_layer = previous_layer, activation_function = OUTPUT_ACTIVATION_FUNCTION, initial_value = 0.0, **kwargs)
        self.set_previous_layer(previous_layer)

        weights = []
        for item in self:
            weights.append(item.get_weights_layer())

        self.set_weights_layer(weights)

# NEURAL NETWORK
class NeuralNetwork():
    def __init__(self, neurons_per_hidden_layer, input_layer_size, output_layer_size):
        self.input_layer = InputLayer(input_layer_size)

        self.hidden_layers = []

        previous_layer = self.input_layer
        for k in range(len(neurons_per_hidden_layer)):
            self.hidden_layers.append(NeuronLayer(previous_layer, neurons_per_hidden_layer[k]))
            previous_layer = self.hidden_layers[k]

        self.output_layer = OutputLayer(previous_layer, output_layer_size)

    def update_all_layers(self):
        for hl in nn.hidden_layers:
            hl.update_layer()
        nn.output_layer.update_layer()
        
# TEST CODE

nn = NeuralNetwork([4, 2], 2, 1)
nn.input_layer[0].set_value(0)
nn.input_layer[1].set_value(32)

nn.update_all_layers()

print nn.hidden_layers[0]
"""
sigmoid = Sigmoid()
iL  = InputLayer(2, initial_value=sigmoid(3.0))
hL1 = NeuronLayer(iL, 4)
hL2 = NeuronLayer(hL1, 2, initial_value = 1.0)
oL  = OutputLayer(hL2, 1)

print iL
print hL1
print hL2
print oL

hL1.update_layer()
hL2.update_layer()
oL.update_layer()

print '--- --- ---'
print iL
print hL1
print hL2
print oL

w = hL1.get_weights_layer()
w[2][1].set_value(0.01)

hL1.update_layer()
hL2.update_layer()
oL.update_layer()

print '--- --- ---'
print iL
print hL1
print hL2
print oL

print '=== === ==='
print hL1.get_weights_layer()
print hL2.get_weights_layer()
print oL.get_weights_layer()
"""
