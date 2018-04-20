from nn import NeuralNetwork

# TEST CODE

nn = NeuralNetwork([4, 2], 2, 1)
nn.input_layer[0].set_value(0)
nn.input_layer[1].set_value(2)

nn.update_all_layers()

print nn.hidden_layers[0]
