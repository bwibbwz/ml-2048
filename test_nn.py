from nn import NeuralNetwork
from random import randint

# TEST CODE

nn = NeuralNetwork([4, 2], 2, 1)
nn.input_layer.set_values([0, 2])

nn.update_all_layers()
print nn.input_layer, nn.output_layer

for k in range(10):
    in1 = nn.input_layer.get_values()[0]
    in2 = nn.input_layer.get_values()[1]
    new_in = [in1, in2]
    if in1 == in2:
        if randint(0, 1) == 0:
            new_in = [in1 + in2, 0]
        else:
            new_in = [0, in1 + in2]
    elif in1 == 0:
        new_in[0] = 2
    elif in2 == 0:
        new_in[1] = 2
    elif in1 < in2:
        new_in[0] *=2
    elif in1 > in2:
        new_in[1] *=2
        
    nn.input_and_update(new_in)
    print nn.input_layer, nn.output_layer


