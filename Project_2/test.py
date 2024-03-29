import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from deap import base
from deap import creator
from deap import tools
import random

from itertools import chain
from operator import attrgetter

from deap.benchmarks.tools import  hypervolume

################################################################################################ Functions ################################################################################################
def check_feasiblity(individual):
    '''
    Feasibility function for the individual. 
    Returns True if individual is feasible (or constraint not violated),
    False otherwise
    '''
    if (len(set(individual)) != len(individual)): # Verify if indiviual contains repeated cities
        return False
    else:
        return True

# Crossover operator
def Greedy_crossover(ind1, ind2):
    ''' 
    Greedy crossover function produces two new individuals.
    Determine a city d as a starting point of crossover and set it as the first genetic point in both new individuals.
    To generate the first new individual find the cities to the right of d (dr1 and dr2 from individual 1 and 2) and calculate the distances 
    (d, dr1) and (d,dr2), j1 and j2 respectively if j1<j2 put dr1 as the next genetic point and set d = dr1
    otherwise put dr2 as the next point and set d = dr2, in both cases delete the previous d from the parents
    then iterate this procedure until there's only one element left each parent parent.
    To generate the second individual a similar process is used, but instead of the right cities we use left cities
    and j1 and j2 are the travel cost from d to dl1 and dl2 respectively.
    '''
    # We must keep the original indiviuals somewhere to use later on
    temp1, temp2 = ind1.copy(), ind2.copy()
    temp3, temp4 = ind1.copy(), ind2.copy()

    d = temp1[0] #set d
    ind1, ind2 = [], [] # initialize the new individuals
    ind1.append(d)
    
    while len(temp1)>1:
        try:
            dr1 = temp1[temp1.index(d)+1] 
        except:
            dr1 = temp1[0]

        j1 = cost_distance.iloc[d+1,dr1+1] # get distance between cities of 1st parent 

        try:
            dr2 = temp2[temp2.index(d)+1]
        except:
            dr2 = temp2[0]

        j2 = cost_distance.iloc[d+1,dr2+1] # get distance between cities of 2nd parent 

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

        j1 = travel_cost[d+1,dl1+1] # get traveling cost between cities of 1st parent 

        try:
            dl2 = temp4[temp4.index(d)-1]
        except:
            dl2 = temp4[-1]

        j2 = travel_cost[d+1,dl2+1] # get traveling cost between cities of 2nd parent

        temp3.remove(d)
        temp4.remove(d)

        if j1>=j2: # here we have j1>=j2 because we want to go first to the cities with a greater travel cost
            ind2.append(dl1)
            d=dl1
        else:
            ind2.append(dl2)
            d=dl2

    return ind1, ind2


# Mutation operator    
def Greedy_invert_mutation(ind):
    '''
    Greedy invert mutation function produces a new individual.
    First a city m1 is chosen at random. Then it finds the city m2 with the shortest distance
    to m1 from the cities to the right of m1. Finally switch the position in the chromosome of the city to right 
    of m1 with m2.
    '''
    temp = ind.copy()
    size = len(temp)
    m1 = temp[random.randint(0, size-2)] # set m1
    # if random.random() <=1:
    if random.random() <=0.1:
        cost = 100000
        for element in temp[ind.index(m1)+1:]: # go through cities to the right of m1
            
            if cost_distance.iloc[m1+1,element+1]<cost: # verify if new element has a smaller distance
                cost = cost_distance.iloc[m1+1,element+1]
                m2 = element

        ind[ind.index(m2)], ind[ind.index(m1)+1]  = temp[temp.index(m1)+1], temp[temp.index(m2)]   # position switch
    else:
        cost = 0
        for element in temp[ind.index(m1)+1:]: # go through cities to the right of m1
            
            if travel_cost[m1+1,element+1]>cost: # verify if new element has a larger cost
                cost = travel_cost[m1+1,element+1]
                m2 = element

        ind[ind.index(m2)], ind[ind.index(m1)+1]  = temp[temp.index(m1)+1], temp[temp.index(m2)]   # position switch
    return ind

# Min Cost: (Dist;Cost) = (708.0,377430.0)
# Min Dist: (Dist;Cost) = (698.0,423650.0)  

# Selection operator
def Mod_NSGA2(individuals, k, nd='standard'):
    '''
    Normal NSGA 2 algorithm with a small modification, where instead of using 
    the crowding distance it uses the Euclidean distance.
    '''

    if nd == 'standard':
        pareto_fronts = tools.sortNondominated(individuals, k)
    elif nd == 'log':
        pareto_fronts = tools.sortLogNondominated(individuals, k)
    else:
        raise Exception('selNSGA2: The choice of non-dominated sorting '
                        'method "{0}" is invalid.'.format(nd))

    for front in pareto_fronts:
        assignEuclDist(front) # assign Euclidean distance

    chosen = list(chain(*pareto_fronts[:-1]))
    k = k - len(chosen)
    if k > 0:
        sorted_front = sorted(pareto_fronts[-1], key=attrgetter("fitness.crowding_dist"), reverse=True)
        chosen.extend(sorted_front[:k])

    return chosen

def assignEuclDist(individuals):
    """
    Assign a Euclidean distance to each individual's fitness. The
    Euclidean distance can be retrieve via the :attr:`crowding_dist`
    attribute of each individual's fitness.
    """
    if len(individuals) == 0:
        return

    distances = [0.0] * len(individuals)
    crowd = [(ind.fitness.values, i) for i, ind in enumerate(individuals)]

    crowd.sort(key=lambda element: element[0][1])

    distances[crowd[0][1]] = float("inf")
    distances[crowd[-1][1]] = float("inf")

    for prev, cur, next in zip(crowd[:-2], crowd[1:-1], crowd[2:]):
        distances[cur[1]] = np.linalg.norm(np.array(next[0]) - np.array(cur[0])) + np.linalg.norm(np.array(cur[0]) - np.array(prev[0])) # sum of euclidean distances between current point and its two neighbors
    
    for i, dist in enumerate(distances):
        individuals[i].fitness.crowding_dist = dist

def evaluate(individuals):
    '''
    Evaluate function calculates the fitness values for each individual.
    '''
    # distance and transport cost vectors initialization
    dist_value=[]
    transp_value=[]
    limit_orders = 1000 
    dist_value.append(cost_distance.values[0,individuals[0]+1]) # get distance from one location to the next one
    transp_value.append(cost_distance.values[0,individuals[0]+1] * limit_orders) # get traveling cost from one location to the next one
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
    dist_cost = sum(dist_value) # total distance covered by the individual 
    transp_cost = sum(transp_value) # total traveling cost of the individual 
    return dist_cost, transp_cost

def main(plot_10, plot_30):
    '''
    The main function goes through the whole genetic algorithm that was implemented
    '''
    runs = 1 #number of runs
    min_min=1000000000
    f1_min = []
    for i in range(runs):
        print(i)
        # random.seed(34+i)
        random.seed(64)
        n_pop= 100 # population size
        CXPB = 1 # crossover probability
        NGEN = 100 # number of generations

        g=0 # generation initialization

        
        PF=tools.ParetoFront() # pareto front initialization

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
        

        pop = toolbox.select(pop, len(pop))

        record = stats.compile(pop)
        logbook.record(gen=0, evals=len(invalid_ind), **record) # tracking the statistics of each generation
        print(logbook.stream)

        #pareto front update
        PF.update(population = pop)

        hv = [0]*(NGEN) # initialize hypervolume saving vector

        ref_point = record['max'] # get worst point from first generation and set as reference point for hypervolume calculation
        while g < NGEN-1:
            g = g + 1 

            offspring = tools.selTournamentDCD(pop, len(pop)) # selecting a diversified offspring
            offspring = [toolbox.clone(ind) for ind in offspring]
            
            # Crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values  
            
            # Mutation
            for mutant in offspring:
                toolbox.mutate(mutant)
                del mutant.fitness.values
            
            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            pop = toolbox.select(offspring, len(offspring)) # create new population

            record = stats.compile(pop)
            logbook.record(gen=g, evals=len(invalid_ind), **record)
            print(logbook.stream)

            
            PF.update(population= pop) # pareto front update
            
            hv[g] = hypervolume(pop, ref_point) # save the hypervolume of this generation
        

        pareto_front_values = np.array([ind.fitness.values for ind in PF.items])
        
        # pareto front plot for one run
        # plt.figure()
        # plt.title('Pareto curve')
        # plt.xlabel('Cost')
        # plt.ylabel('Dist')
        # if n_cities == 10: 
        #     plot_10 = plt.scatter(pareto_front_values[:,1],pareto_front_values[:,0], c="r")
        # elif n_cities == 30: 
        #     plot_30 = plt.scatter(pareto_front_values[:,1],pareto_front_values[:,0], c="g")
        # elif n_cities == 50: 
        #     plot_50 = plt.scatter(pareto_front_values[:,1],pareto_front_values[:,0], c="b")

        # Save all pareto fronts from all runs in one image
        # if n_cities==50:
        #     plt.legend([plot_10, plot_30, plot_50],[ '10 cities','30 cities','50 cities'])
        #     # plt.savefig('MOOP\Pareto_curve.png')
       
        # save plots
        # if n_cities == 10:
        #     plt.savefig('MOOP_10\Pareto\pareto10_%s.png' % i)
        # elif n_cities == 30:
        #     plt.savefig('MOOP_30\Pareto\pareto30_%s.png' % i)
        # else:
        #     plt.savefig('MOOP_50\Pareto\pareto50_%s.png' % i)


        # hypervolume evolution for one run
        # plt.figure()
        # plt.plot(hv)
        # plt.title('Hypervolume evolution curve')
        # plt.xlabel('Generations')
        # plt.ylabel('Hypervolume')
        # plt.show()

        # save plots
        # if n_cities == 10:
        #     plt.savefig('MOOP_10\Hypervolume\Hyperv10_%s.png' % i)
        # elif n_cities == 30:
        #     plt.savefig('MOOP_30\Hypervolume\Hyperv30_%s.png' % i)
        # else:
        #     plt.savefig('MOOP_50\Hypervolume\Hyperv50_%s.png' % i)

        #save best population (taking in account the sum of cost and distance normalized)
        f1, f2 = record['min']
        f1_min.append(f1)
        if f1/ref_point[0] + f2/ref_point[1] < min_min:
            pop_best=pop
            min_min = f1/ref_point[0] + f2/ref_point[1]
            hv_best = hv
   

    
    # Plot best pareto solutions and best population
    PF.update(pop_best)
    pareto_front_values = np.array([ind.fitness.values for ind in PF.items])
    # front = np.array([ind.fitness.values for ind in pop_best])
    # plt.title('Pareto curve for the best run')
    # plt.xlabel('Cost')
    # plt.ylabel('Dist')            
    # plt.scatter(front[:,0], front[:,1], c="b")
    # plt.scatter(pareto_front_values[:,1],pareto_front_values[:,0], c="r")
    # plt.show()

    # Plot hypervolume evolution for the best population
    # plt.figure()
    # plt.plot(hv_best)
    # plt.title('Hypervolume evolution curve')
    # plt.xlabel('Generations')
    # plt.ylabel('Hypervolume')
    # if n_cities == 10: 
    #     plot_10 = plt.plot(hv_best, c="r", label='10 cities')
    # elif n_cities == 30: 
    #     plot_30 = plt.plot(hv_best, c="g", label='30 cities')
    # elif n_cities == 50: 
    #     plot_50 = plt.plot(hv_best, c="b", label='50 cities')
    # plt.show()

    # Save Hypervolume evolutions from 3 best populations (one for each case) from all runs in one image
    # if n_cities==50:
    #     plt.legend()
    #     plt.savefig('MOOP\Hypervolume_evolution.png')

    print("-- End of (successful) evolution --")

    min_dist_idx = np.argmin(pareto_front_values[:,0]) # index where distance is minimum
    min_cost_idx = np.argmin(pareto_front_values[:,1]) # index where traveling cost is minimum

    print('\n')
    print('Min Cost: (Dist;Cost) = (%s,%s)' % (pareto_front_values[min_cost_idx,0],pareto_front_values[min_cost_idx,1]))
    print('Min Dist: (Dist;Cost) = (%s,%s)' % (pareto_front_values[min_dist_idx,0],pareto_front_values[min_dist_idx,1]))


    # print(np.mean(f1_min)) # average of minimum distances over all runs 
    return plot_10, plot_30


################################################################################################  MAIN  ################################################################################################
n_cases = 1 # how many cases to test (if 3 it could be 1 test per case (10, 30 and 50))
plot_10, plot_30 = 0,0
for i in range(n_cases):
    # Get user input
    write = True
    while write:
        print('How many cities will the problem have (10, 30 or 50)?\n')
        n_cities = int(input())
        if n_cities != 10 and n_cities != 30 and n_cities != 50:
            print('Error: please choose a number of cities of: 10, 30 or 50 \n')
        else:
            write = False

    # Get problem data
    cost_distance = pd.read_csv('CustDist_WHCentral.csv', header= 0, names =  np.arange(51))
    location = pd.read_csv('CustXY_WHCentral.csv')
    orders = pd.read_csv('CustOrd.csv')

    # Generate travel cost matrix
    a = cost_distance.iloc[:n_cities+1,:n_cities+1].to_numpy()
    b = orders["Orders"].iloc[:n_cities+1].to_numpy()*np.identity(n_cities+1)
    travel_cost = np.matmul(a, b)

    ################################  PROBLEM OBJECTIVE  ################################

    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    ################################  POPULATION  ################################

    toolbox = base.Toolbox()
    toolbox.register('Genes', random.sample, range(0,n_cities), n_cities)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.Genes)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    ################################ OPERATORS ################################

    toolbox.register("evaluate", evaluate)
    toolbox.decorate("evaluate", tools.DeltaPenalty(check_feasiblity, 10000)) 
    toolbox.register("mate", Greedy_crossover)
    toolbox.register("mutate", Greedy_invert_mutation)  
    toolbox.register("select", Mod_NSGA2)

    ################################ MINIMIZATION ################################

    plot_10, plot_30 = main(plot_10, plot_30)