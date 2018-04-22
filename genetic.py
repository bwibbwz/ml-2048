from random import random
from nn import RandomWrapper, NeuralNetwork

class GeneticAlgorithm(list):
    def __init__(self, generation_size, **kwargs):
        self.individual_parameters = kwargs
        self.generation_size = generation_size

    def add_new_generation(self):
        super(GeneticAlgorithm, self).append(Generation(self.generation_size, **self.individual_parameters))
        for individual in self[-1]:
            individual.set_all_weights(self.generate_random_weights())

    def __repr__(self):
        return str([individual.get_fitness() for individual in self[-1]])

    def generate_random_weights(self):
        shape = self[0].get_weights_shape()
        weights = []
        for layer in shape:
            weights.append([])
            for sub in layer:
                weights[-1].append([random() for _ in range(sub)])
        return weights

class Individual(NeuralNetwork):
    def __init__(self, fitness=0.0, **kwargs):
        super(Individual, self).__init__(**kwargs)
        self.fitness = fitness

    def set_fitness(self, fitness):
        self.fitness = fitness

    def get_fitness(self):
        return self.fitness

class Generation(list):
    def __init__(self, size, **kwargs):
        self.individual_parameters = kwargs
        self.individuals = []
        for k in range(size):
            super(Generation, self).append(Individual(fitness = 0.0, **self.individual_parameters))
        
    def set_random_fitness(self):
        for ind in self:
            ind.set_fitness(random())

    def get_weights_shape(self):
        return self[0].get_weights_shape()

    def sort_by_fitness(self):
        self.sort(key=lambda ind: ind.get_fitness())

    def __repr__(self):
        return str([ind.get_fitness() for ind in self])
        
