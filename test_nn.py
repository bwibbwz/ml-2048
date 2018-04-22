from nn import NeuralNetwork
from random import randint
from gamepython.run import run2048
from activation_functions import DiscreteAF, TanH, PassThrough, Log2, ReLU


# TEST CODE

import numpy as np
rgame = run2048(True, False, 0, 1)
nn = NeuralNetwork([30, 30], 17, 1, input_af = Log2(),
                   hidden_af = [ReLU(), ReLU()],
                   output_af = DiscreteAF(3, TanH))

for i in range(13):
    if i == 0:
        matrix, check = rgame.get_status()
        inp = np.array(matrix).flatten().tolist()
        print check
        inp.append(check)

        nn.input_and_update(inp)
        print nn.output_layer.get_values()
    else:
        rgame.run(input_value=nn.output_layer.get_values()[0])
        matrix, check = rgame.get_status()
        inp = np.array(rgame.gamegrid.matrix).flatten().tolist()
        inp.append(check)
        print check
        nn.input_and_update(inp)
        print nn.output_layer.get_values()


#for k in range(10):
#    new_in = input_values
#    in1 = new_in[0]
#    in2 = new_in[1]
#    if in1 == in2:
#        if randint(0, 1) == 0:
#            new_in = [in1 + in2, 0]
#        else:
#            new_in = [0, in1 + in2]
#    elif in1 == 0:
#        new_in[0] = 2
#    elif in2 == 0:
#        new_in[1] = 2
#    elif in1 < in2:
#        new_in[0] *=2
#    elif in1 > in2:
#        new_in[1] *=2
#        
#    input_values = new_in
#    nn.input_and_update(new_in)
#    print input_values, nn.input_layer, nn.output_layer
#
#print '-- -'
#from genetic import GeneticAlgorithm
#ga = GeneticAlgorithm(generation_size = 4, neurons_per_hidden_layer = [4, 2], input_layer_size = 2, output_layer_size = 1, input_af = Log2(), hidden_af = [ReLU(), ReLU()], output_af = DiscreteAF(2, TanH))
#ga.add_new_generation()
#
#for gen in ga:
#    print ' === Generation ==='
#    for ind in gen:
#        print ' --- Individual ---'
#        for layer in ind:
#            print layer
#        print ind.get_all_weights()
#        print ' --- Individual ---'
#    gen.set_random_fitness()
#    print "Fitness: ", gen
#    print "Fitness: ", gen.sort()
#    print "Fitness: ", gen
#    print ' === Generation ==='

