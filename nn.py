from random import random
from activation_functions import *

# OTHER FUNCTIONS
class RandomWrapper():
    def __call__(self):
        return random()*0.03

# ACTIVATION FUNCTION DEFAULTS
NEURON_ACTIVATION_FUNCTION = TanH()
WEIGHT_ACTIVATION_FUNCTION = PassThrough()
NODE_ACTIVATION_FUNCTION = PassThrough()
INPUT_ACTIVATION_FUNCTION = PassThrough()
OUTPUT_ACTIVATION_FUNCTION = DiscreteAF(4, TanH)

# OTHER DEFULATS
WEIGHT_INITIAL_VALUE = RandomWrapper()

# NODES, NEURONS AND WEIGHTS
class Node(object):
    def __init__(self, initial_input_value=None, initial_output_value=None, bias=0.0, activation_function=NODE_ACTIVATION_FUNCTION):
        self.set_activation_function(activation_function)
        self.set_input_value(initial_input_value)
        self.set_output_value(initial_output_value)
        self.set_bias(bias)

    def set_activation_function(self, activation_function):
        self.activation_function = activation_function

    def get_bias(self):
        return self.bias

    def set_bias(self, bias):
        self.bias = bias

    def get_input_value(self):
        return self.input_value

    def set_input_value(self, input_value):
        self.input_value = input_value

    def get_output_value(self):
        return self.output_value

    def set_output_value(self, output_value):
        self.output_value = output_value

    def update_value(self):
        self.set_output_value(self.activation_function(self.get_bias() + self.get_input_value()))

    # NB: This is here just for completeness against the previous code.
    def get_value(self):
        return self.get_output_value()

    # NB: This is here just for completeness against the previous code.
    def set_value(self, value):
        self.set_input_value(value)

    def __repr__(self):
        return str(self.get_output_value())

class Neuron(Node):
    def __init__(self, previous_layer, activation_function=NEURON_ACTIVATION_FUNCTION):
        super(Neuron, self).__init__(activation_function = activation_function)
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
            new_value += pl[k].get_output_value() * wl[k].get_value()
        self.set_input_value(new_value)
        self.set_output_value(self.activation_function(self.get_bias() + self.get_input_value()))

class Weight(Node):
    def __init__(self, node_in, node_out, initial_value=WEIGHT_INITIAL_VALUE):
        self.node_in = node_in
        self.node_out = node_out
        super(Weight, self).__init__(initial_input_value = initial_value, activation_function = WEIGHT_ACTIVATION_FUNCTION)
        self.update_value()

# LAYERS
class Layer(list):
    def __init__(self, number_of_items, item_class=Node, **kwargs):
        for k in range(number_of_items):
            super(Layer, self).append(item_class(**kwargs))
        self.set_previous_layer(None)
        self.set_weights_layer(None)

    def set_values(self, values):
        if len(values) != len(self):
            raise ValueError("The length of the input values (%i) does not match the length of the layer (%i)." % (len(values), len(self)))
        else:
            for k in range(len(self)):
                self[k].set_input_value(values[k])
                self[k].update_value()

    def get_values(self):
        return [node.get_output_value() for node in self]

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

    def get_weights_shape(self):
        return [len(w) for w in self.weights]

class WeightLayer(Layer):
    def __init__(self, node, previous_layer, number_of_items, item_class=Weight, **kwargs):
        for k in range(number_of_items):
            super(Layer, self).append(item_class(previous_layer[k], node, initial_value = WEIGHT_INITIAL_VALUE()))
        self.set_previous_layer(previous_layer)
    
    def update_layer(self):
        pass

    def get_weights_shape(self):
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
    def __init__(self, number_of_items, item_class=Node, activation_function=INPUT_ACTIVATION_FUNCTION, **kwargs):
        super(InputLayer, self).__init__(number_of_items, item_class, activation_function = activation_function, **kwargs)
        self.set_previous_layer(None)

#    def update_layer(self):
#        pass
        
    def get_weights_shape(self):
        pass

class OutputLayer(Layer):
    def __init__(self, previous_layer, number_of_items, item_class=Neuron, activation_function=OUTPUT_ACTIVATION_FUNCTION, **kwargs):
        super(OutputLayer, self).__init__(number_of_items, item_class, previous_layer = previous_layer, activation_function = activation_function, **kwargs)
        self.set_previous_layer(previous_layer)

        weights = []
        for item in self:
            weights.append(item.get_weights_layer())

        self.set_weights_layer(weights)

# NEURAL NETWORK
class NeuralNetwork(object):
    def __init__(self, neurons_per_hidden_layer, input_layer_size, output_layer_size, input_af=None, hidden_af=None, output_af=None):
        
        if input_af is None:
            self.input_layer = InputLayer(input_layer_size)
        else:
            self.input_layer = InputLayer(input_layer_size, activation_function = input_af)

        self.hidden_layers = []

        # BUG: Need to check the dimensions of neurons_per_hidden_layer
        # BUG: Need to check the dimensions of hidden_af further

        previous_layer = self.input_layer
        for k in range(len(neurons_per_hidden_layer)):
            if hidden_af is None:
                self.hidden_layers.append(NeuronLayer(previous_layer, neurons_per_hidden_layer[k]))
            elif len(hidden_af) == len(neurons_per_hidden_layer):
                self.hidden_layers.append(NeuronLayer(previous_layer, neurons_per_hidden_layer[k], activation_function = hidden_af[k]))
            elif len(hidden_af) == 1:
                self.hidden_layers.append(NeuronLayer(previous_layer, neurons_per_hidden_layer[k], activation_function = hidden_af[0]))
            else:
                raise ValueError('The dimensions of hidden_af (%i) do not match the amount of hidden layers (%i).' % (len(hidden_af), len(neurons_per_hidden_layer)))
            previous_layer = self.hidden_layers[k]

        self.output_layer = OutputLayer(previous_layer, output_layer_size)
        if output_af is None:
            self.output_layer = OutputLayer(previous_layer, output_layer_size)
        else:
            self.output_layer = OutputLayer(previous_layer, output_layer_size, activation_function = output_af)

    def update_all_layers(self):
        for hl in self.hidden_layers:
            hl.update_layer()
        self.output_layer.update_layer()

    def input_values(self, input_values):
        self.input_layer.set_values(input_values)

    def input_and_update(self, input_values):
        self.input_values(input_values)
        self.update_all_layers()
        return self.output_layer.get_values()

    def get_weights_shape(self):
        shape = []
        for layer in self:
            layer_shape = layer.get_weights_shape()
            if layer_shape is not None:
                shape.append(layer_shape)
        return shape

    def get_all_weights(self):
        all_weights = []
        for layer in self:
            weights_layer = layer.get_weights_layer()
            if weights_layer is not None:
                all_weights.append(weights_layer)
        return all_weights

    #NB: Need better checks for the array shapes.
    def set_all_weights(self, new_weights):
        for k in range(len(self) - 1):
            layer = self[k + 1]
            weights = layer.get_weights_layer()
            for h in range(len(weights)):
                weights[h] = new_weights[k][h]

    def __len__(self):
        return len(self.hidden_layers) + 2

    def __iter__(self):
        yield self.input_layer
        for hL in self.hidden_layers:
            yield hL
        yield self.output_layer

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if index < 0 or index > len(self) - 1:
            raise IndexError("The item being referenced, is outside the size of the NeuralNetwork.")
        if index == 0:
            return self.input_layer
        elif index == len(self) - 1:
            return self.output_layer
        else:
            return self.hidden_layers[index - 1]
            
