import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from deap import base
from deap import creator
from deap import tools
import random
from more_itertools import one

def plot_costumer_location(xy, max_client):
    
    fig, ax = plt.subplots()
    ax.scatter(xy['X'][0:max_client],  xy['Y'][0:max_client])
    ax.scatter(xy['X'][0], xy['Y'][0], c = '#d62728' , label = "Warehouse")
    
    for i, txt in enumerate(xy['Customer XY'][0:max_client]):
        ax.annotate(txt, (xy['X'][i], xy['Y'][i]))
    
    plt.show()
        
    return

def penalty_fxn(individual):

    return pow(int(evaluate(individual)[0]),2)

def check_feasiblity(individual):
    '''
    Feasibility function for the individual. 
    Returns True if individual is feasible (or constraint not violated),
    False otherwise
    '''
    #if ((sum(costumers)) != sum(individual)):
    if (len(set(individual)) != len(individual)):
        # Indiviual contains repeated values
        # Indiviual contains repeated values
        return False
    else:
        return True

def evaluate(individuals):
    value=[]
    limit_orders = 1000
    value.append(cost_distance.values[0,individuals[0]+1])
    for i in range(n_ind-1):
        limit_orders = limit_orders - orders['Orders'][individuals[i]+1] 
        if orders['Orders'][individuals[i+1]+1] > limit_orders :
            value.append(cost_distance.values[individuals[i]+1,0])
            value.append(cost_distance.values[0,individuals[i+1]+1])
            limit_orders = 1000
        value.append(cost_distance.values[individuals[i]+1,individuals[i+1]+1])
    value.append(cost_distance.values[individuals[n_ind-1]+1,0])
    fit_ind = (sum(value))
    return fit_ind,

def minimization():

    random.seed(64)
    n_pop=100
    CXPB , MUTPB = 0.6, 0.3
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
   
    print("-- End of (successful) evolution --")
    best_ind = tools.selBest(pop, 1)[0]
    best_ind1 = [x+1 for x in best_ind]
    print("Best individual is %s, %s" % (best_ind1, best_ind.fitness.values)) 
    
    return


##################################  MAIN  ############################################

n_ind = 30

cost_distance = pd.read_csv('CustDist_WHCentral.csv', header= 0, names =  np.arange(51))
location = pd.read_csv('CustXY_WHCentral.csv')
orders = pd.read_csv('CustOrd.csv')
zeros = np.zeros(1, dtype=int)
plot_costumer_location(location, n_ind+1)

################################  MINIMIZATION CLASS  ###################################

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

###############################  CREATING POPULATION  ######################################

toolbox = base.Toolbox()
toolbox.register('Genes', random.sample, range(0,n_ind), n_ind )
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.Genes)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate)
toolbox.decorate("evaluate", tools.DeltaPenalty(check_feasiblity, 10000)) 
toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)  
toolbox.register("select", tools.selTournament, tournsize=4)


minimization()
