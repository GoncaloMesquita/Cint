import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from deap import base
from deap import creator
from deap import tools
import random
from more_itertools import one

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
n_ind = 51
n = np.arange(n_ind)
toolbox = base.Toolbox()
toolbox.register('Genes', random.sample, range(n_ind),n_ind )
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.Genes)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate(individuals):
    value=[]
    value.append(cost_distance.values[0,individuals[0]])
    for i in range(n_ind-2):
        value.append(cost_distance.values[individuals[i],individuals[i+1]])
    value.append(cost_distance.values[individuals[n_ind-1],0])
    fit_ind = (sum(value))
    return fit_ind,

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint )
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.7)  
toolbox.register("select", tools.selTournament, tournsize=4)

def minimization(distance):
    random.seed(64)
    n=100
    CXPB , MUTPB = 0.6, 0.4
    pop = toolbox.population(100)
    print(pop)
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    g=0
    while g < 50:
    
        g = g + 1

        offspring = toolbox.select(pop, len(pop))  
        offspring = list(map(toolbox.clone, offspring))

        # CROSSOVER

        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values 
        
        #Mutation

        for mutant in offspring:

            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        #Revaluate the ones wich were mutated or crossovered (invalid fitness)

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)

        #Update the fitness with the offsprings

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            print(ind.fitness.values[0])


        pop[:] = offspring



    

    return


##################################  MAIN  ############################################

n = np.arange(n_ind)

cost_distance = pd.read_csv('CustDist_WHCentral.csv', header= 0, names= n)
location = pd.read_csv('CustXY_WHCentral.csv')
orders = pd.read_csv('CustOrd.csv')
#print(cost_distance)
""" print(location)
print(orderes)  """

plt.scatter(x=location['X'][1:], y=location['Y'][1:], marker='x')
plt.scatter(x=location['X'][0], y=location['Y'][0])
plt.show()
minimization(cost_distance)
