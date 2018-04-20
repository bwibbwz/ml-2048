from nn import NeuralNetwork

# TEST CODE

nn = NeuralNetwork([4, 2], 2, 1)
nn.input_layer[0].set_value(0)
nn.input_layer[1].set_value(2)

nn.update_all_layers()
print nn.output_layer

nn.input_layer.set_values([1.0, 1.5])
nn.update_all_layers()
print nn.output_layer
