from nn import NeuralNetwork
from random import randint
from activation_functions import DiscreteAF, TanH, PassThrough, Log2


# TEST CODE

nn = NeuralNetwork([4, 2], 2, 1, input_af = Log2(), hidden_af = [TanH(), TanH()], output_af = DiscreteAF(2, TanH))
input_values = [0, 2]
nn.input_and_update(input_values)

print input_values, nn.input_layer, nn.output_layer

for k in range(10):
    new_in = input_values
    in1 = new_in[0]
    in2 = new_in[1]
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
        
    input_values = new_in
    nn.input_and_update(new_in)
    print input_values, nn.input_layer, nn.output_layer


