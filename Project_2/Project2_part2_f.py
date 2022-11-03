import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from deap import base
from deap import creator
from deap import tools
import random

from deap.benchmarks.tools import diversity, convergence, hypervolume


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
    if (len(set(individual)) != len(individual)):
        # Indiviual contains repeated values
        return False
    else:
        return True

def greedy_crossover(ind1, ind2):
    # We must keep the original values somewhere before scrambling everything
    temp1, temp2 = ind1.copy(), ind2.copy()
    temp3, temp4 = ind1.copy(), ind2.copy()

    d = temp1[0]
    ind1, ind2 = [], []
    ind1.append(d)
    
    while len(temp1)>1:
        try:
            dr1 = temp1[temp1.index(d)+1] 
        except:
            dr1 = temp1[0]
        j1 = cost_distance.iloc[d+1,dr1+1]
        try:
            dr2 = temp2[temp2.index(d)+1]
        except:
            dr2 = temp2[0]
        j2 = cost_distance.iloc[d+1,dr2+1]

        temp1.remove(d)
        temp2.remove(d)

        if j1<=j2:
            ind1.append(dr1)
            d=dr1
        else:
            ind1.append(dr2)
            d=dr2

    d = temp3[0]
    ind2.append(d)
    while len(temp3)>1:
        
        try:
            dl1 = temp3[temp3.index(d)-1]
        except:
            dl1 = temp3[-1]

        j1 = travel_cost[d+1,dl1+1]
        try:
            dl2 = temp4[temp4.index(d)-1]
                  
        except:
            dl2 = temp4[-1]

        j2 = travel_cost[d+1,dl2+1]

        temp3.remove(d)
        temp4.remove(d)
        if j1<=j2:
            ind2.append(dl1)
            d=dl1
        else:
            ind2.append(dl2)
            d=dl2

    return ind1, ind2
    
def invert_mutation(ind):
    temp = ind.copy()
    size = len(temp)
    m1 = temp[random.randint(0, size-2)]
    cost = 100000
    for element in temp[ind.index(m1)+1:]:
        if cost_distance.iloc[m1+1,element+1]<cost: 
            cost = cost_distance.iloc[m1+1,element+1]
            m2 = element

    ind[ind.index(m2)], ind[ind.index(m1)+1]  = temp[temp.index(m1)+1], temp[temp.index(m2)] 

    
    # dist, transp = evaluate(ind)
    # dist1, transp1 = evaluate(temp)
    # if (dist > dist1) and (transp > transp1):
    #     ind = temp.copy()
    # print(ind)
    return ind



def evaluate(individuals):
    # print(individuals)
    dist_value=[]
    transp_value=[]
    limit_orders = 1000
    dist_value.append(cost_distance.values[0,individuals[0]+1])
    transp_value.append(cost_distance.values[0,individuals[0]+1] * limit_orders)
    for i in range(n_cities-1):
        limit_orders = limit_orders - orders['Orders'][individuals[i]+1] 
        if orders['Orders'][individuals[i+1]+1] > limit_orders :
            dist_value.append(cost_distance.values[individuals[i]+1,0])
            transp_value.append(cost_distance.values[individuals[i]+1,0] * limit_orders)
            dist_value.append(cost_distance.values[0,individuals[i+1]+1])
            limit_orders = 1000
            transp_value.append(cost_distance.values[0,individuals[i+1]+1] * limit_orders)
        dist_value.append(cost_distance.values[individuals[i]+1,individuals[i+1]+1])
        transp_value.append(cost_distance.values[individuals[i]+1,individuals[i+1]+1] * limit_orders)
    dist_value.append(cost_distance.values[individuals[n_cities-1]+1,0])
    transp_value.append(cost_distance.values[individuals[n_cities-1]+1,0] * limit_orders)
    dist_cost = sum(dist_value)
    transp_cost = sum(transp_value)
    return dist_cost, transp_cost

def minimization():
    runs = 1 #number of runs
    min_avg=1000000000
    for i in range(runs):
        random.seed(64)

        # random.seed(41)
        # n_pop=4
        n_pop=100
        # CXPB = 0.6
        CXPB = 1
        # NGEN = 2
        NGEN = 100

        

        PF=tools.ParetoFront()

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
        
        g=0

        pop = toolbox.select(pop, len(pop))

        record = stats.compile(pop)
        logbook.record(gen=0, evals=len(invalid_ind), **record)
        print(logbook.stream)

        #pareto front update
        PF.update(population= pop)

        hv = [None]*(NGEN+1) #init hypervolume saving vector

        while g < NGEN-1:
            
            g = g + 1
            # print('Genration ', g)
            # offspring = toolbox.select(pop, len(pop))  
            # offspring = list(map(toolbox.clone, offspring))
            # print('\n\n')
            # print(pop)
            offspring = tools.selTournamentDCD(pop, len(pop))

            # print(offspring)
            # print('\n\n')
            offspring = [toolbox.clone(ind) for ind in offspring]
            # print(offspring)

            
            # CROSSOVER
            for child1, child2 in zip(offspring[::2], offspring[1::2]):

                if random.random() < CXPB:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values  

            
            #Mutation
            

            for mutant in offspring:

                # if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

            
            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # print(pop)
            # print(pop[0].fitness)
            # print(offspring)
            # print(offspring[0].fitness)
            # print('\n\n')


            pop = toolbox.select(offspring, len(pop))
            # pop = toolbox.select(pop + offspring, len(pop))
            # print(pop)

            record = stats.compile(pop)
            logbook.record(gen=g, evals=len(invalid_ind), **record)
            print(logbook.stream)


            #pareto front update
            PF.update(population= pop) 

            #save the hypervolume of this generation
            hv[g] = hypervolume(pop, [3069, 1870500])
        
            PF.update(pop)

            # pareto_front_values = np.array([ind.fitness.values for ind in PF.items])
            # plt.title('Pareto curve for the best run')
            # plt.xlabel('Cost')
            # plt.ylabel('Dist')
            # front = np.array([ind.fitness.values for ind in pop])
                    
            # plt.scatter(front[:,0], front[:,1], c="b")
            # plt.scatter(pareto_front_values[:,0],pareto_front_values[:,1], c="r")
            # plt.show()
        #save best population (taking in account the sum of Cost and Time)
        f1, f2 = record['avg']

        if f1+f2 < min_avg:
            pop_best=pop
            min_avg=f1+f2
            hv_best = hv

    PF.update(pop_best)

    pareto_front_values = np.array([ind.fitness.values for ind in PF.items])
    plt.title('Pareto curve for the best run')
    plt.xlabel('Cost')
    plt.ylabel('Dist')
    # front = np.array([ind.fitness.values for ind in pop_best])
            
    # plt.scatter(front[:,0], front[:,1], c="b")
    plt.scatter(pareto_front_values[:,1],pareto_front_values[:,0], c="r")
    # plt.show()

    print("-- End of (successful) evolution --")
    # best_ind = tools.selBest(pop, 1)[0]
    # best_ind1 = [x+1 for x in best_ind]
    # print("Best individual is %s, %s" % (best_ind1, best_ind.fitness.values)) 

    min_dist_idx = np.argmin(pareto_front_values[:,0])
    min_cost_idx = np.argmin(pareto_front_values[:,1])

    print('\n')
    print('Min Cost: (Dist;Cost) = (%s,%s)' % (pareto_front_values[min_cost_idx,0],pareto_front_values[min_cost_idx,1]))
    print('Min Dist: (Dist;Cost) = (%s,%s)' % (pareto_front_values[min_dist_idx,0],pareto_front_values[min_dist_idx,1]))

    #plot hypervolume evolution for the best population
    plt.figure()
    plt.plot(hv_best)
    # plt.plot(hv)
    plt.title('Hypervolume curve for the best run')
    plt.xlabel('Generations')
    plt.ylabel('Hypervolume')
    plt.show()
    return


##################################  MAIN  ############################################
write = True
while write:
    print('How many cities will the problem have (10, 30 or 50)?\n')
    n_cities = int(input())
    if n_cities != 10 and n_cities != 30 and n_cities != 50:
        print('Error: please choose a number of cities of: 10, 30 or 50 \n')
    else:
        write = False


cost_distance = pd.read_csv('CustDist_WHCentral.csv', header= 0, names =  np.arange(51))
location = pd.read_csv('CustXY_WHCentral.csv')
orders = pd.read_csv('CustOrd.csv')
# plot_costumer_location(location, n_cities+1)
a = cost_distance.iloc[:n_cities+1,:n_cities+1].to_numpy()
b = orders["Orders"].iloc[:n_cities+1].to_numpy()*np.identity(n_cities+1)
travel_cost = np.matmul(a, b)
# print(a)
# print(b)
# print(np.matmul(a, b))

################################  MINIMIZATION CLASS  ###################################

creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMin)

###############################  CREATING POPULATION  ######################################

toolbox = base.Toolbox()
toolbox.register('Genes', random.sample, range(0,n_cities), n_cities )
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.Genes)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate)
toolbox.decorate("evaluate", tools.DeltaPenalty(check_feasiblity, 10000)) 
toolbox.register("mate", greedy_crossover)
toolbox.register("mutate", invert_mutation)  
toolbox.register("select", tools.selNSGA2)




minimization()

## Tools selNGA2
# Min Cost: (Dist;Cost) = (306.0,222120.0)
# Min Dist: (Dist;Cost) = (306.0,222120.0)
# 314.26666666666665

# Min Cost: (Dist;Cost) = (648.0,330560.0)
# Min Dist: (Dist;Cost) = (603.0,332780.0)
# 711.1

# Min Cost: (Dist;Cost) = (1003.0,487360.0)
# Min Dist: (Dist;Cost) = (1003.0,487360.0)
# 1087.9

## Modified NGA2
# Min Cost: (Dist;Cost) = (306.0,222120.0)
# Min Dist: (Dist;Cost) = (306.0,222120.0)
# 312.96666666666664 

# Min Cost: (Dist;Cost) = (617.0,312820.0)
# Min Dist: (Dist;Cost) = (588.0,314990.0)
# 698.5333333333333

# Min Cost: (Dist;Cost) = (931.0,452860.0)
# Min Dist: (Dist;Cost) = (927.0,453020.0)
# 1103.4