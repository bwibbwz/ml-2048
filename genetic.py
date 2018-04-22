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

    def populate_new_generation(self, parents, generation, carry_on_top_parents=0, add_random=0, mix_odds=0.0, mutate_odds=0.0):
        parents.sort(reverse=True)
        parent_weights = [parent.get_all_weights() for parent in parents]
        child_weights = []

        for k in range(add_random):
            parent_weights.append(generate_random_weights)
        index = range(len(parent_weights))

        while len(index) > 1 and len(child_weights) < len(generation):
            male = index.pop(randint(0, len(index) - 1))
            female = index.pop(randint(0, len(index) - 1))
            child = self.breed_weights(parent_weights[male], parent_weights[female], mix_odds = mix_odds, mutate_odds = mutate_odds)
            child_weights.append(child)

        if len(index) == 1 and len(child_weights) < len(generation):
            male = index.pop(randint(0, len(index) - 1))
            female = randint(0, len(parent_weights))
            child = self.breed_weights(parent_weights[male], parent_weights[female], mix_odds = mix_odds, mutate_odds = mutate_odds)
            child_weights.append(child)

        # NB: Should I add mutate_odds here?
        for k in range(carry_on_top_parents): 
            child_weights.append(parent_weights[k])

        while len(child_weights) < len(generation):
            male = randint(0, len(parent_weights) - 1)
            female = randint(0, len(parent_weights) - 1)
            child = self.breed_weights(parent_weights[male], parent_weights[female], mix_odds = mix_odds, mutate_odds = mutate_odds)
            child_weights.append(child)

        for k, individual in enumerate(generation):
            individual.set_all_weights(child_weights[k])

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

    def breed_weights(self, male_weights, female_weights, mix_odds=0.0, mutate_odds=0.0):
        child_weights = []
        shape = self[0].get_weights_shape()
        for k in range(len(shape)):
            child_weights.append([])
            for h in range(len(shape[k])):
                child_weights[-1].append([])
                for j in range(shape[k][h]):
                    new_weight = 0.0
                    if random() < mix_odds:
                        new_weight = (male_weights[k][h][j] + female_weights[k][h][j]) / 2.0
                    elif random() < mutate_odds:
                        new_weight = random()
                    elif randint(0,1) == 0:
                        new_weight = male_weights[k][h][j]
                    else:
                        new_weight = female_weights[k][h][j]
                    child_weights[-1][-1].append(new_weight)
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
        
