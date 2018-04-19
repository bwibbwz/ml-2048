from random import random, randint

# ACTIVATION FUNCTIONS
class ActivationFunction():
    def __init__(self):
        pass

class PassThrough(ActivationFunction):
    def transform_value(self, input_value):
        return input_value

class RelU(ActivationFunction):
    def transform_value(self, input_value):
        return max(0, input_value)

# SIMPLE VALUE CONSTRUCTS
class BasicFloat():
    def __init__(self, initial_value):
        self.value = initial_value

    def set_value(self, value):
        print '---> ', value, self.activation_function(value)
        self.value = self.activation_function(value)

    def get_value(self):
        return self.value

    def __repr__(self):
        return str(self.get_value())

    def set_activation_function(self, activation_function=PassThrough):
        self.activation_function = activation_function

class Neuron(BasicFloat):
    def __init__(self, initial_value=0.0):
        BasicFloat.__init__(self, initial_value)

class Weight(BasicFloat):
    def __init__(self, initial_value=None):
        if initial_value == None:
            BasicFloat.__init__(self, random())
        else:
            BasicFloat.__init__(self, initial_value)

# LAYER CONSTRUCTS
class Layer():
    def __init__(self, item_class=None, number_of_items=1, **kwargs):
        if item_class is None:
            self.items = [None]*number_of_items
        else:
            self.items = []
            for k in range(number_of_items):
                self.items.append(item_class(**kwargs))
 
        self.previous_layer = None
        self.weights_layer = None

    def update_layer_values(self):
        if self.get_previous_layer() is None or self.get_weights_layer() is None:
            raise RunTimeError('Either a previous layer or its associated weights have not been defined.')
        else:
            for k in range(len(self)):
                new_value = 0.0
                for h in range(len(self.previous_layer)):
                    new_value += self.previous_layer.get_value(h) * self.weights_layer.get_value(h,k)
                self.set_value(k, new_value)

    def get_previous_layer(self):
        return self.previous_layer

    def get_weights_layer(self):
        return self.weights_layer

    def set_previous_layer(self, previous_layer):
        self.previous_layer = previous_layer

    def set_weights_layer(self, weights_layer):
        self.weights_layer = weights_layer

    def get_value(self, index):
        return self.items[index]

    def set_value(self, index, value):
        self.items[index] = value

    def set_all_values(self, values):
        if len(values) != len(self):
            raise ValueError("Length of the values to be set (%i) must be equal to the size of the layer (%i)." % (len(values), len(self)))
        for k in range(len(values)):
            self.set_value(k, values[k])

    def initialize_all_values(self, function, *argv):
        for k in range(len(self)):
            Layer.set_value(self, k, function(*argv))

    def __repr__(self):
        returner = ''
        for k in range(len(self.items)):
            returner += self.items[k].__repr__() + '\n\r'
        return returner[:-2]

    def __len__(self):
        return len(self.items)

class NeuronLayer(Layer):
    def __init__(self, previous_layer, number_of_neurons, initial_value=0.0):
        Layer.__init__(self, Neuron, number_of_neurons, initial_value = initial_value)
        self.previous_layer = previous_layer
        self.weight_layer = WeightLayer(previous_layer, self)

class InputLayer(Layer):
    def __init__(self, number_of_input_nodes):
        Layer.__init__(self, BasicFloat, number_of_input_nodes, initial_value = None)

class WeightLayer(Layer):
    def __init__(self, layer1, layer2, initial_values=None):
        Layer.__init__(self, Weight, len(layer1) * len(layer2))

        # Create the mapping between each node/neuron of the previous layer with the nodes/neurons of the current layer
        self.layers_to_weight_map = []
        k = 0
        for j in range(len(layer1)):
            self.layers_to_weight_map.append([])
            for h in range(len(layer2)):
                self.layers_to_weight_map[j].append(k)
                k += 1

    def set_value(self, l1i, l2i, value):
        self.items[self.layers_to_weight_map[l1i][l2i]] = value
        
    def get_value(self, l1i, l2i):
        return self.items[self.layers_to_weight_map[l1i][l2i]]
        
class OutputLayer(Layer):
    def __init__(self, previous_layer, number_of_output_nodes):
        Layer.__init__(self, BasicFloat, number_of_output_nodes, initial_value = None)
        
        self.set_previous_layer(previous_layer)
        weights_layer = WeightLayer(previous_layer, self)
        self.set_weights_layer(weights_layer)

## --- --- ---
i1 = InputLayer(4)
print(i1)
i1.set_all_values([1.0, 3, 4.0, "A"])
print(i1)

l1 = NeuronLayer(i1, 4)
l1.set_all_values([1.1, 1.2, 1.3, 1.4])
print(l1)
l2 = NeuronLayer(l1, 5)
l2.set_all_values([0.1, -0.2, -0.3, 0.4, 0.5])
print(l2)

print '-- -- -- --'
o1 = OutputLayer(l2, 2)
print o1
wo1 = o1.get_weights_layer()
wo1.initialize_all_values(random)

print '-- -- -- --'
o1.update_layer_values()
print 'asdf', o1

l2.set_value(1, 1.0)
o1.update_layer_values()
print 'asdf', o1

