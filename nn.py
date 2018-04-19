from random import random, randint

# SIMPLE VALUE CONSTRUCTS
class BasicFloat():
    def __init__(self, initial_value):
        self.value = initial_value

    def set_value(self, value):
        self.value = value

    def get_value(self):
        return self.value

    def __repr__(self):
        return str(self.get_value())

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
    def __init__(self, item_class, number_of_items, **kwargs):
        self.items = []
        for k in range(number_of_items):
            self.items.append(item_class(**kwargs))

    def set_value(self, index, value):
        self.items[index] = value

    def set_all_values(self, values):
        if len(values) != len(self):
            raise ValueError("Length of the values to be set (%i) must be equal to the size of the layer (%i)." % (len(values), len(self)))
        for k in range(len(values)):
            self.set_value(k, values[k])

    def __repr__(self):
        returner = ''
        for k in range(len(self.items)):
            returner += self.items[k].__repr__() + '\n\r'
        return returner[:-2]

    def __len__(self):
        return len(self.items)

class NeuronLayer(Layer):
    def __init__(self, number_of_neurons, initial_value=0.0):
        Layer.__init__(self, Neuron, number_of_neurons, initial_value = initial_value)

class InputLayer(Layer):
    def __init__(self, number_of_inputs):
        self.items = [None]*number_of_inputs

class WeightLayer(Layer):
    def __init__(self, layer1, layer2, initial_values=None):
        Layer.__init__(self, Weight, len(layer1) * len(layer2))
        self.layers_to_weight_map = []
        k = 0
        for j in range(len(layer1)):
            self.layers_to_weight_map.append([])
            for h in range(len(layer2)):
                self.layers_to_weight_map[j].append(k)
                if j > h:
                    self.set_value(j, h, j + h)
                else:
                    self.set_value(j, h, j * h)
                k += 1

    def set_value(self, l1i, l2i, value):
        self.items[self.layers_to_weight_map[l1i][l2i]] = value
        
    def get_value(self, l1i, l2i):
        return self.items[self.layers_to_weight_map[l1i][l2i]]
        

## --- --- ---
n1 = Neuron(1.5)
print(n1.get_value())
n1.set_value(3)
print(n1.get_value())

l1 = NeuronLayer(4)
print(l1)
l2 = NeuronLayer(5)
print(l2)

i1 = InputLayer(4)
print(i1)
i1.set_all_values([1.0, 3, 4.0, "A"])
print(i1)

wl = WeightLayer(l1, l2)
print wl.get_value(3,4)


