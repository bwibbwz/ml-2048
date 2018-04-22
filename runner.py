from genetic import Individual
from random import randint
from gamepython.run import run2048
from activation_functions import DiscreteAF, TanH, PassThrough, Log2, ReLU
import numpy as np

class Runner(object):
    def __init__(self, game, individual, print_steps=False):
        self.game = game
        self.print_steps = print_steps
        self.individual = individual
        self.fitness_penalty = 0

    def step(self):
        if self.calculate_fitness < 0:
            # NYI: Stop game
            pass
        game = self.game
        individual = self.individual
        matrix, repeat_check = game.get_status()
        game_state = np.array(game.gamegrid.matrix).flatten().tolist()
        game_state.append(repeat_check)
        game_input = individual.input_and_update(game_state)
        game.run(input_value = game_input[0])
        self.fitness_penalty -= repeat_check
        if self.print_steps:
            print "State: %s | %i | Fitness: %i" % (matrix, repeat_check, self.calculate_fitness())

    def calculate_fitness(self):
        score = max(np.array(self.game.gamegrid.matrix).flatten().tolist())
        penalty = self.fitness_penalty
        return score + penalty
        
game = run2048(manual_input = True, random = False, steps = 0, sleep = 1)
ind = Individual(neurons_per_hidden_layer = [20, 20],
                 input_layer_size = 17,
                 output_layer_size = 1,
                 input_af = Log2(),
                 hidden_af = [TanH(), TanH()],
                 output_af = DiscreteAF(4, TanH))

runner = Runner(game, ind, print_steps = True)
for k in range(13):
    runner.step()
    print "Fitness: ",  runner.calculate_fitness()

