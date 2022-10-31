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

def greedy_crossover(ind1, ind2):
    # We must keep the original values somewhere before scrambling everything

    temp1, temp2 = ind1.copy(), ind2.copy()
    temp3, temp4 = ind1.copy(), ind2.copy()

    # print(evaluate(temp1))

    d = temp1[0]
    ind1, ind2 = [], []
    ind1.append(d)
    
    while len(temp1)>1:
        # j1 = 1000000
        # j2 = 1000000
        # print(temp1)
        # print(temp2)
        try:
            dr1 = temp1[temp1.index(d)+1] 
        except:
            dr1 = temp1[0]

        j1 = cost_distance.iloc[d+1,dr1+1]
        # print(j1)


        try:
            dr2 = temp2[temp2.index(d)+1]
        except:
            dr2 = temp2[0]

        j2 = cost_distance.iloc[d+1,dr2+1]
        # print(j2)

        temp1.remove(d)
        temp2.remove(d)

        if j1<=j2:
            ind1.append(dr1)
            d=dr1
        else:
            ind1.append(dr2)
            d=dr2

    # print(evaluate(ind1))
    # print(evaluate(temp4))


    d = temp3[0]
    ind2.append(d)
    while len(temp3)>1:
        # j1 = 1000000
        # j2 = 1000000
        # print(temp3)
        # print(temp4)
        # print(ind2)
        try:
            dl1 = temp3[temp3.index(d)-1]
            
        except:
            dl1 = temp3[-1]

        j1 = cost_distance.iloc[d+1,dl1+1]
        # j1 = travel_cost[d+1,dl1+1]
        # print(j1)


        try:
            dl2 = temp4[temp4.index(d)-1]
                  
        except:
            dl2 = temp4[-1]

        j2 = cost_distance.iloc[d+1,dl2+1]
        # j2 = travel_cost[d+1,dl2+1]
        # print(j2)

        temp3.remove(d)
        temp4.remove(d)

        if j1<=j2:
            ind2.append(dl1)
            d=dl1
        else:
            ind2.append(dl2)
            d=dl2
    # print(evaluate(ind2))
    # print(1)

    # print(ind1)
    # print(ind2)




 

    return ind1, ind2

def evaluate(individuals):
    dist_value=[]
    transp_value=[]
    limit_orders = 1000
    dist_value.append(cost_distance.values[0,individuals[0]+1])
    transp_value.append(cost_distance.values[0,individuals[0]+1] * limit_orders)
    for i in range(n_ind-1):
        limit_orders = limit_orders - orders['Orders'][individuals[i]+1] 
        if orders['Orders'][individuals[i+1]+1] > limit_orders :
            dist_value.append(cost_distance.values[individuals[i]+1,0])
            transp_value.append(cost_distance.values[individuals[i]+1,0] * limit_orders)
            dist_value.append(cost_distance.values[0,individuals[i+1]+1])
            limit_orders = 1000
            transp_value.append(cost_distance.values[0,individuals[i+1]+1] * limit_orders)
        dist_value.append(cost_distance.values[individuals[i]+1,individuals[i+1]+1])
        transp_value.append(cost_distance.values[individuals[i]+1,individuals[i+1]+1] * limit_orders)
    dist_value.append(cost_distance.values[individuals[n_ind-1]+1,0])
    transp_value.append(cost_distance.values[individuals[n_ind-1]+1,0] * limit_orders)
    dist_cost = sum(dist_value)
    transp_cost = sum(transp_value)
    return dist_cost,

def minimization():

    random.seed(64)
    n_pop=100
    CXPB , MUTPB = 0.6, 0.3


    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("avg", np.mean, axis=0)
    stats.register("max", np.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    pop = toolbox.population(n_pop)
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    print(logbook.stream)
    
    g=0
    while g < 100:
        
        g = g + 1
        # print('Genration ', g)
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
            
        # pop[:] = offspring
        pop = toolbox.select(pop + offspring, len(pop))

        record = stats.compile(pop)
        logbook.record(gen=g, evals=len(invalid_ind), **record)
        print(logbook.stream)
        # fits = [ind.fitness.values[0] for ind in pop]
        
        # length = len(pop)
        # mean = sum(fits) / length
        # sum2 = sum(x*x for x in fits)
        # std = abs(sum2 / length - mean**2)**0.5
    
        # print("  Min %s" % min(fits))
        # print("  Max %s" % max(fits))
        # print("  Avg %s" % mean)
        # print("  Std %s" % std) 
   
    print("-- End of (successful) evolution --")
    best_ind = tools.selBest(pop, 1)[0]
    best_ind1 = [x+1 for x in best_ind]
    print("Best individual is %s, %s" % (best_ind1, best_ind.fitness.values)) 
    
    return


##################################  MAIN  ############################################

n_ind = 50

cost_distance = pd.read_csv('CustDist_WHCentral.csv', header= 0, names =  np.arange(51))
location = pd.read_csv('CustXY_WHCentral.csv')
orders = pd.read_csv('CustOrd.csv')
zeros = np.zeros(1, dtype=int)
# plot_costumer_location(location, n_ind+1)

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
toolbox.register("mate", greedy_crossover)
# toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)  
toolbox.register("select", tools.selTournament, tournsize=4)


minimization()
