from random import random, randint
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

    def breed_weights(self, male_weights, female_weights):
        child_weights = []
        shape = self[0].get_weights_shape()
        for k in range(len(shape)):
            child_weights.append([])
            for h in range(len(shape[k])):
                child_weights[-1].append([])
                for j in range(shape[k][h]):
                    if randint(0,1) == 0:
                        child_weights[-1][-1].append(male_weights[k][h][j])
                    else:
                        child_weights[-1][-1].append(female_weights[k][h][j])
        return child_weights

class Individual(NeuralNetwork):
    def __init__(self, fitness=0.0, **kwargs):
        super(Individual, self).__init__(**kwargs)
        self.fitness = fitness

    def set_fitness(self, fitness):
        self.fitness = fitness

    def get_fitness(self):
        return self.fitness

    def __eq__(self, other):
        if hasattr(other, 'fitness'):
            return self.get_fitness() == other.get_fitness()

    def __ne__(self, other):
        if hasattr(other, 'fitness'):
            return self.get_fitness() != other.get_fitness()

    def __gt__(self, other):
        if hasattr(other, 'fitness'):
            return self.get_fitness() > other.get_fitness()

    def __ge__(self, other):
        if hasattr(other, 'fitness'):
            return self.get_fitness() >= other.get_fitness()

    def __lt__(self, other):
        if hasattr(other, 'fitness'):
            return self.get_fitness() < other.get_fitness()

    def __le__(self, other):
        if hasattr(other, 'fitness'):
            return self.get_fitness() <= other.get_fitness()

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

    def __repr__(self):
        return str([ind.get_fitness() for ind in self])
        
