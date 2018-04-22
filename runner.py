from genetic import Individual
from random import randint
from gamepython.run import run2048
from gamepython.logic import game_state
from activation_functions import DiscreteAF, TanH, PassThrough, Log2, ReLU, Scale
import numpy as np

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
        if self.has_game_ended(matrix):
            return False
        for neuron in individual.input_layer:
            neuron.set_activation_function(Scale(1.0/max(game_state)))
        game_input = individual.input_and_update(game_state)
        game.run(input_value = game_input[0])
        self.fitness_penalty -= repeat_check
        if self.print_steps:
            print "Input: %i | State: %s | %i | Fitness: %i" % (game_input[0], matrix, repeat_check, self.calculate_fitness())
        return True

    def has_game_ended(self, matrix):
        return self.calculate_fitness() < 0 or game_state(matrix) == 'lose'

    def calculate_fitness(self):
        score = max(np.array(self.game.gamegrid.matrix).flatten().tolist())
        penalty = self.fitness_penalty
        return score + penalty
        
game = run2048(manual_input = True, random = False, steps = 0, sleep = 0)
ind = Individual(neurons_per_hidden_layer = [10, 10],
                 input_layer_size = 17,
                 output_layer_size = 1,
                 input_af = Log2(),
                 hidden_af = [ReLU(), ReLU()],
                 output_af = DiscreteAF(4, TanH))

runner = Runner(game, ind, print_steps = True)
while runner.step():
    pass
game.gamegrid.destroy()



