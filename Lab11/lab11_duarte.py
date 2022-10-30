#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.

import array
import random
import json

import matplotlib.pyplot as plt
import numpy as np

from math import sqrt

from deap import algorithms
from deap import base
from deap import benchmarks
from deap.benchmarks.tools import diversity, convergence, hypervolume
from deap import creator
from deap import tools

# creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
# creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)

creator.create("FitnessMax", base.Fitness, weights=(1.0, -1.0))
creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMax)

toolbox = base.Toolbox()



BOUND_LOW, BOUND_UP = 0.0, 1

NDIM = 2

def uniform(low, up, size=None):
    try:
        return [random.uniform(a, b) for a, b in zip(low, up)]
    except TypeError:
        return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]

def evalTwoMax(individual):
    Z1 = np.sqrt(individual[0]**2 + individual[1]**2)
    Z2 = np.sqrt((individual[0]-1)**2 + (individual[1]+1)**2)
    f1 = np.sin(4*Z1)/Z1 + np.sin(2.5*Z2)/Z2
    f2 = 1 - np.sin(5*Z1)/Z1
    return f1, f2

toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
# toolbox.register("attr_float", random.uniform, 0, 1)
# toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, 2)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

#toolbox.register("evaluate", benchmarks.zdt1)
toolbox.register("evaluate", evalTwoMax)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM)
toolbox.register("select", tools.selNSGA2)

# toolbox.register("attr_float", random.uniform, 0, 10)
# toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, 2)
# toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# toolbox.register("evaluate", evalTwoMax)
# toolbox.register("mate", tools.cxOnePoint)
# toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=20, indpb=0.2)
# toolbox.register("select", tools.selBest)

def main(seed=None):
    random.seed(seed)

    NGEN = 20
    MU = 100
    CXPB = 0.9

    # Pareto Front
    PF = tools.ParetoFront()

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("avg", np.mean, axis=0)
    stats.register("max", np.max, axis=0)
    
    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"
    
    pop = toolbox.population(n=MU)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # This is just to assign the crowding distance to the individuals
    # no actual selection is done
    pop = toolbox.select(pop, len(pop))
    
    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    print(logbook.stream)

    #pareto front update
    PF.update(population= pop)
    

    plt.figure()
    # Begin the generational process
    for gen in range(1, NGEN):
        # Vary the population
        # offspring = tools.selTournament(pop, len(pop), 3)
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]
        
        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= CXPB:
                toolbox.mate(ind1, ind2)
            
            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
            del ind1.fitness.values, ind2.fitness.values
        
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select the next generation population
        pop = toolbox.select(pop + offspring, MU)
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        print(logbook.stream)

        #pareto front update
        PF.update(population= pop) 

        pareto_front_values = np.array([ind.fitness.values for ind in PF.items])
        # print(pareto_front_values)
        front = np.array([ind.fitness.values for ind in pop])
                
        plt.scatter(front[:,0], front[:,1], c="b")
        plt.scatter(pareto_front_values[:,0],pareto_front_values[:,1], c="r")
        plt.show()

    print("Final population hypervolume is %f" % hypervolume(pop, [11.0, 11.0]))

    
    # pareto_front_values = np.array([ind.fitness.values for ind in PF.items])

    # plt.figure()
    # plt.scatter(pareto_front_values[:,0],pareto_front_values[:,1])
    # plt.show()

    return pop, logbook
        
if __name__ == "__main__":

    
    pop, stats = main()
    # pop.sort(key=lambda x: x.fitness.values)
    
    # front = np.array([ind.fitness.values for ind in pop])
    # print(PF)
    # # print(stats)
    # pf_x = [ind[0] for ind in PF]
    # pf_y = [ind[1] for ind in PF]


    # plt.scatter(pf_x, pf_y, c="r")
    # plt.scatter(front[:,0], front[:,1], c="b")
    
    # # front = np.array([ind.fitness.values for ind in pop])
    # # plt.scatter(front[:,0], front[:,1], c="b")
    # plt.axis("tight")
    # plt.show()


























