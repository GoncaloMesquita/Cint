# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from deap import base
# from deap import creator
# from deap import tools
# import random

# from deap.benchmarks.tools import diversity, convergence, hypervolume


# def plot_costumer_location(xy, max_client):
    
#     fig, ax = plt.subplots()
#     ax.scatter(xy['X'][0:max_client],  xy['Y'][0:max_client])
#     ax.scatter(xy['X'][0], xy['Y'][0], c = '#d62728' , label = "Warehouse")
    
#     for i, txt in enumerate(xy['Customer XY'][0:max_client]):
#         ax.annotate(txt, (xy['X'][i], xy['Y'][i]))
    
#     plt.show()
        
#     return


# def check_feasiblity(individual):
#     '''
#     Feasibility function for the individual. 
#     Returns True if individual is feasible (or constraint not violated),
#     False otherwise
#     '''
#     if (len(set(individual)) != len(individual)):
#         # Indiviual contains repeated values
#         return False
#     else:
#         return True

# def evaluate(individuals):
#     dist_value=[]
#     transp_value=[]
#     limit_orders = 1000
#     dist_value.append(cost_distance.values[0,individuals[0]+1])
#     transp_value.append(cost_distance.values[0,individuals[0]+1] * limit_orders)
#     for i in range(n_ind-1):
#         limit_orders = limit_orders - orders['Orders'][individuals[i]+1] 
#         if orders['Orders'][individuals[i+1]+1] > limit_orders :
#             dist_value.append(cost_distance.values[individuals[i]+1,0])
#             transp_value.append(cost_distance.values[individuals[i]+1,0] * limit_orders)
#             dist_value.append(cost_distance.values[0,individuals[i+1]+1])
#             limit_orders = 1000
#             transp_value.append(cost_distance.values[0,individuals[i+1]+1] * limit_orders)
#         dist_value.append(cost_distance.values[individuals[i]+1,individuals[i+1]+1])
#         transp_value.append(cost_distance.values[individuals[i]+1,individuals[i+1]+1] * limit_orders)
#     dist_value.append(cost_distance.values[individuals[n_ind-1]+1,0])
#     transp_value.append(cost_distance.values[individuals[n_ind-1]+1,0] * limit_orders)
#     dist_cost = sum(dist_value)
#     transp_cost = sum(transp_value)
#     return dist_cost, transp_cost

# def minimization():

#     random.seed(64)
#     n_pop=100
#     CXPB , MUTPB = 0.6, 0.3


    

#     PF=tools.ParetoFront()

#     stats = tools.Statistics(lambda ind: ind.fitness.values)
#     stats.register("std", np.std, axis=0)
#     stats.register("min", np.min, axis=0)
#     stats.register("avg", np.mean, axis=0)
#     stats.register("max", np.max, axis=0)

#     logbook = tools.Logbook()
#     logbook.header = "gen", "evals", "std", "min", "avg", "max"




#     pop = toolbox.population(n_pop)

#     # Evaluate the individuals with an invalid fitness
#     invalid_ind = [ind for ind in pop if not ind.fitness.valid]
#     fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
#     for ind, fit in zip(invalid_ind, fitnesses):
#         ind.fitness.values = fit
    
#     g=0

#     pop = toolbox.select(pop, len(pop))

#     record = stats.compile(pop)
#     logbook.record(gen=0, evals=len(invalid_ind), **record)
#     print(logbook.stream)

#     #pareto front update
#     PF.update(population= pop)

#     # while g < 1:
#     while g < 100:
        
#         g = g + 1
#         # print('Genration ', g)
#         # offspring = toolbox.select(pop, len(pop))  
#         # offspring = list(map(toolbox.clone, offspring))

#         offspring = tools.selTournamentDCD(pop, len(pop))
#         offspring = [toolbox.clone(ind) for ind in offspring]

#         # CROSSOVER

#         for child1, child2 in zip(offspring[::2], offspring[1::2]):

#             if random.random() < CXPB:
#                 toolbox.mate(child1, child2)
#                 del child1.fitness.values
#                 del child2.fitness.values  
        
#         #Mutation

#         for mutant in offspring:

#             if random.random() < MUTPB:
#                 toolbox.mutate(mutant)
#                 del mutant.fitness.values

#         # Evaluate the individuals with an invalid fitness
#         invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
#         fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
#         for ind, fit in zip(invalid_ind, fitnesses):
#             ind.fitness.values = fit

#         # pop[:] = offspring
            
#         pop = toolbox.select(pop + offspring, len(pop))

#         record = stats.compile(pop)
#         logbook.record(gen=g, evals=len(invalid_ind), **record)
#         print(logbook.stream)

#         # fits = [ind.fitness.values[0] for ind in pop]
        
#         # length = len(pop)
#         # mean = sum(fits) / length
#         # sum2 = sum(x*x for x in fits)
#         # std = abs(sum2 / length - mean**2)**0.5
    
#         # print("  Min %s" % min(fits))
#         # print("  Max %s" % max(fits))
#         # print("  Avg %s" % mean)
#         # print("  Std %s" % std) 

#         #pareto front update
#         PF.update(population= pop) 

#     pareto_front_values = np.array([ind.fitness.values for ind in PF.items])
#     # print(pareto_front_values)
#     # front = np.array([ind.fitness.values for ind in pop])
            
#     # plt.scatter(front[:,0], front[:,1], c="b")
#     plt.scatter(pareto_front_values[:,0],pareto_front_values[:,1], c="r")
#     plt.show()
   
#     print("-- End of (successful) evolution --")
#     best_ind = tools.selBest(pop, 1)[0]
#     print(best_ind)
#     best_ind1 = [x+1 for x in best_ind]
#     print("Best individual is %s, %s" % (best_ind1, best_ind.fitness.values)) 
    
#     return


# ##################################  MAIN  ############################################

# n_ind = 50

# cost_distance = pd.read_csv('CustDist_WHCentral.csv', header= 0, names =  np.arange(51))
# location = pd.read_csv('CustXY_WHCentral.csv')
# orders = pd.read_csv('CustOrd.csv')
# # plot_costumer_location(location, n_ind+1)

# ################################  MINIMIZATION CLASS  ###################################

# creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
# creator.create("Individual", list, fitness=creator.FitnessMin)

# ###############################  CREATING POPULATION  ######################################

# toolbox = base.Toolbox()
# toolbox.register('Genes', random.sample, range(0,n_ind), n_ind )
# toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.Genes)
# toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# toolbox.register("evaluate", evaluate)
# toolbox.decorate("evaluate", tools.DeltaPenalty(check_feasiblity, 10000)) 
# toolbox.register("mate", tools.cxOrdered)
# toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)  
# toolbox.register("select", tools.selNSGA2)


# minimization()

