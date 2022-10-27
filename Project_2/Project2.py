import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from deap import base
from deap import creator
from deap import tools
import random
from more_itertools import one


def minimization():

    random.seed(64)
    n_pop=100
    CXPB , MUTPB = 0.6, 0.6
    pop = toolbox.population(n_pop)
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    g=0
    while g < 100:
        
        g = g + 1
        print('Genration ', g)
        offspring = toolbox.select(pop, len(pop))  
        offspring = list(map(toolbox.clone, offspring))

        # CROSSOVER
        """ 
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values 
         """
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
            
        pop[:] = offspring

        fits = [ind.fitness.values[0] for ind in pop]
        
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
    
        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std) 
        #fit = np.array(fits)
        #best_10 = fit.argsort()
        #print("-- End of (successful) evolution --")
        best_ind = tools.selBest(pop, 1)[0]
        print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values)) 

    count = 0
    cost=[]
    for i in (best_ind):
        if i == 0 :
            cost.append(count)
            count = 0
        count = orders['Orders'][i]+count
    cost.append(count)
    print("Contagem  ida", cost )
        

    

    return

def evaluate(individuals):
    value=[]
    value.append(cost_distance.values[0,individuals[0]])
    for i in range(n_ind):
        value.append(cost_distance.values[individuals[i],individuals[i+1]])
    value.append(cost_distance.values[individuals[n_ind],0])
    fit_ind = (sum(value))
    return fit_ind,

##################################  MAIN  ############################################
n_ind = 51
indiv = np.arange(n_ind)

cost_distance = pd.read_csv('CustDist_WHCentral.csv', header= 0, names = indiv)
location = pd.read_csv('CustXY_WHCentral.csv')
orders = pd.read_csv('CustOrd.csv')
zeros = np.zeros(1, dtype=int)
costumers = np.append(indiv,zeros)

plt.scatter(x=location['X'][1:], y=location['Y'][1:], marker='x')
plt.scatter(x=location['X'][0], y=location['Y'][0])
plt.show()

################################  MINIMIZATION CLASS  ###################################

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

###############################  CREATING POPULATION  ######################################

toolbox = base.Toolbox()
toolbox.register('Genes', np.random.permutation, costumers )
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.Genes)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxPartialyMatched)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.7)  
toolbox.register("select", tools.selTournament, tournsize=4)


minimization()
