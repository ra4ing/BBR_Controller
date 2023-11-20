import numpy
import random

import numpy as np


class GA:
    # DEFINE here the 3 GA Parameters:
    num_generations = 200
    num_population = 50
    num_elite = 20
    cp = 30
    cpp = 20
    mp = 20

    @staticmethod
    def population_reproduce(genotypes):
        # crossover rate (integer number between 0 and 100):

        genotypes_not_ranked = genotypes
        # Rank: lowest to highest fitness
        genotypes = GA.__rank_population(genotypes)
        # Initiate loop to create new population
        population_size = len(genotypes)
        new_population = []
        # Backwards: from highest to lowest fitness
        for individual in range(population_size, 0, -1):
            # Clone the elite individuals
            if population_size - individual < GA.num_elite:
                new_population.append(genotypes[individual - 1][0])
            elif random.randint(1, 100) < GA.cp:
                parent1 = GA.__select_parent(genotypes_not_ranked)
                parent2 = GA.__select_parent(genotypes_not_ranked)
                # Apply crossover
                child = GA.__crossover(parent1, parent2)
                offspring = GA.__mutation(child)
                new_population.append(numpy.array(offspring))
            else:
                # Generate the rest of the population by using the genetic operations
                parent1 = GA.__select_parent(genotypes)
                # parent2 = GA.__select_parent(genotypes_not_ranked)
                # Apply crossover
                # child = GA.__crossover(parent1, parent2)
                # Apply mutation
                offspring = GA.__mutation(parent1[0])
                new_population.append(numpy.array(offspring))

        # state1_0 = np.load("../pre_module/state1_0.npy")
        # new_population[-1] = state1_0
        return new_population

    @staticmethod
    def __rank_population(genotypes):
        # Rank the populations using the fitness values (Lowest to Highest)
        genotypes.sort(key=lambda item: item[1])
        return genotypes

    @staticmethod
    def get_best_genotype(genotypes):
        return GA.__rank_population(genotypes)[-1]

    @staticmethod
    def get_average_genotype(genotypes):
        summary = 0.0
        for g in range(0, len(genotypes) - 1):
            summary = summary + genotypes[g][1]
        return summary / len(genotypes)

    @staticmethod
    def __select_parent(genotypes):
        # Tournament Selection
        # Select a few individuals of the population randomly
        # group = []
        # # population_size = len(genotypes)
        # number_individuals = 3
        # for selected in range(0, number_individuals - 1):
        #     group.append()
        # # Then, select the best individual of this group
        # group_ranked = GA.__rank_population(group)
        return genotypes[random.choice([0, GA.num_elite])]

    @staticmethod
    def __crossover(parent1, parent2):
        child = []
        # Center
        for gene in range(len(parent1[0])):
            # The new offspring will have its first half of its genes taken from one parent
            if random.randint(1, 100) < GA.cpp:
                child.append(parent1[0][gene])
            else:
                child.append(parent2[0][gene])

        return child

    @staticmethod
    def __mutation(child):
        # Changes a single gene randomly
        after_mutation = []

        # mutation percentage (integer number between 0 and 100):

        for gene in range(len(child)):
            if random.randint(1, 100) < GA.mp:
                # The random value to be added to the gene
                random_value = numpy.random.uniform(-1.0, 1.0, 1)
                temp = child[gene] + random_value[0]
                # Clip
                if temp < -1:
                    temp = -1
                elif temp > 1:
                    temp = 1
                after_mutation.append(temp)
            else:
                after_mutation.append(child[gene])
        return after_mutation

    @staticmethod
    def create_random_population(num_weights):
        #  Size of the population and genotype
        pop_size = (GA.num_population, num_weights)
        # Create the initial population with random weights
        population = np.random.uniform(low=-1, high=1.0, size=pop_size)

        state1_0 = np.load("../pre_module/state1_0.npy")
        right_reach_goal = np.load("../pre_module/right_reach.npy")
        left_reach_goal = np.load("../pre_module/left_reach.npy")
        population[0] = state1_0
        population[1] = state1_0
        population[2] = state1_0
        population[3] = state1_0
        population[4] = state1_0
        population[5] = right_reach_goal
        population[6] = right_reach_goal
        population[7] = right_reach_goal
        population[8] = right_reach_goal
        population[9] = right_reach_goal
        population[10] = left_reach_goal
        population[11] = left_reach_goal
        population[12] = left_reach_goal
        population[13] = left_reach_goal
        population[14] = left_reach_goal

        # for i in range(20):
        #     tmp = np.load("../module/Best{}.npy".format(i))
        #     population[15 + i] = tmp

        return population
