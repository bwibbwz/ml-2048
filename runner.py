from genetic import Individual, GeneticAlgorithm
from random import randint
from gamepython.run import run2048
from gamepython.logic import game_state
from activation_functions import DiscreteAF, TanH, PassThrough, Log2, ReLU, Scale, Sigmoid, SoftMax
import numpy as np
from math import exp

class Runner(object):
    def __init__(self, game, individual, print_steps=False):
        self.game = game
        self.print_steps = print_steps
        self.individual = individual
        self.fitness_penalty = 0

    def step(self):
        game = self.game
        individual = self.individual
        matrix, repeat_check = game.get_status()
        game_state = np.array(game.gamegrid.matrix).flatten().tolist()
        game_state.append(repeat_check)
        if self.has_game_ended(matrix, repeat_check):
            return False
        binary_game_state = self.binarize_input(game_state, binary_size = BINARY_SIZE)
        game_input = self.output_to_move(individual.input_and_update(binary_game_state, output_softmax = True))
#        self.display_values(individual)
        game.run(input_value = game_input)
        self.fitness_penalty -= repeat_check
        if self.print_steps:
            print "Input: %i | State: %s | %i | Fitness: %i" % (game_input, matrix, repeat_check, self.calculate_fitness())
        print individual
        return True

    # Change a list to binary values of length 'binary_size'. The 2^0 digit is omitted since it never appears in the 2048 game.
    def binarize_input(self, game_state, binary_size=11):
        bin_game_state = []
        for number in game_state:
            bin_string = ("{0:0%ib}" % (binary_size + 1)).format(number)[:-1]
            bin_int_list = [int(n) for n in bin_string]
            bin_game_state.extend(bin_int_list)
        return bin_game_state

    def output_to_move(self, nn_output):
        game_move = 0
        max_value = 0.0
        for k in range(len(nn_output)):
           if nn_output[k] > max_value:
                game_move = k
                max_value = nn_output[k] 
        if self.print_steps:
            print '[%.3f %.3f %.3f %.3f]' % tuple(nn_output),
        return game_move

    def has_game_ended(self, matrix, repeat_check):
        return repeat_check > 2 or self.calculate_fitness() < 0 or game_state(matrix) == 'lose'

    def calculate_fitness(self):
        score_max = max(np.array(self.game.gamegrid.matrix).flatten().tolist())
        score_sum = sum(np.array(self.game.gamegrid.matrix).flatten().tolist())
        penalty = self.fitness_penalty
        return score_max + score_sum + penalty
        
GENERATION_SIZE = 4
GENRATION_COUNT = 2
PRINT_STEPS = True
BINARY_SIZE = 4
WEIGHTS_METHOD = 'random'

nn_parameters = {'neurons_per_hidden_layer': [17 * BINARY_SIZE, 17 * BINARY_SIZE, 17 * BINARY_SIZE],
                 'input_layer_size': 17 * BINARY_SIZE,
                 'output_layer_size': 4,
                 'input_af': PassThrough(),
                 'hidden_af':  [TanH(), ReLU(), Sigmoid()],
                 'output_af': TanH()}

game_parameters = {'manual_input': True,
                   'random': False,
                   'steps': 0,
                   'sleep': 0}

ga = GeneticAlgorithm(generation_size = GENERATION_SIZE, **nn_parameters)
ga.add_new_generation(weights_method = WEIGHTS_METHOD)
ga.populate_new_generation(ga[0], ga[0], weights_method = WEIGHTS_METHOD)

for k in range(GENRATION_COUNT):
    print ' --- Generation: %5i ---' % k
    for individual in ga[-1]:
        game = run2048(**game_parameters)
        runner = Runner(game, individual, print_steps = PRINT_STEPS)
        while runner.step():
            pass
        game.gamegrid.destroy()
        individual.set_fitness(runner.calculate_fitness())
#        print individual.get_all_weights()
        print ' === Individual: %5i ===' % individual.get_fitness()

    ga[-1].sort()
    ga.add_new_generation(weights_method = WEIGHTS_METHOD)
    ga.populate_new_generation(ga[-2][:3], ga[-1], weights_method = WEIGHTS_METHOD)

