from json import tool
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from deap import base
from deap import creator
from deap import tools
import random
from functools import partial

def get_parameters():
    
    #Get the number of customers
    write = True
    while write:
        print('How many cities will the problem have ? (10, 30 or 50)')
        n_cost = int(input())
        if n_cost != 10 and n_cost != 30 and n_cost != 50:
            print('Error: please choose a number of cities of: 10, 30 or 50 \n')
        else:
            write = False
    write = True

    #Get the location of the customers/warehouse and distance between them
    while write:
        print('Do you want the warehouse to be centered or in the corner ? (center or corner)')
        ware = str(input())
        if ware != 'center' and ware != 'corner':
            print('Error: please choose a ware postion: corner or center\n')
        else:
            if ware == 'center':
                cost_dis = pd.read_csv('CustDist_WHCentral.csv', header= 0, names =  np.arange(51))
                loc = pd.read_csv('CustXY_WHCentral.csv')
            elif ware == 'corner':
                cost_dis = pd.read_csv('CustDist_WHCorner.csv', header= 0, names =  np.arange(51))
                loc = pd.read_csv('CustXY_WHCorner.csv')
            write = False

    #Get the orders for each customer
    write = True
    while write:
        print('How many orders should we consider? (file or fixed) ')
        t_order = str(input())
        if t_order != 'file' and t_order != 'fixed':
            print('Error: please choose only from: file or fixed \n')
        else:
            if t_order == 'file':
                orders_n = pd.read_csv('CustOrd.csv')
            elif t_order == 'fixed':
                orders_n = pd.read_csv('CustOrd.csv')
                orders_n['Orders']=50
            write = False
    write = True

    #Use heuristic or not 
    while write:
        print('Do you want to use heuristic? (yes or no) ')
        heu = str(input())
        if heu != 'yes' and heu != 'no':
            print('Error: please choose only : yes or no  \n')
        else:   
            write = False

    # Changing the number of tournaments for the selection taking into account the number of individuals

    if n_cost == 10: 
        n_tourn = 4
    if n_cost == 30 :
        n_tourn = 9
    if n_cost == 50:
        n_tourn = 22
    return n_cost, ware, t_order, heu, n_tourn, cost_dis, loc, orders_n

def check_feasiblity(individual):
    '''
    Feasibility function for the individual. 
    Returns True if individual is feasible (or constraint not violated),
    False otherwise
    '''
    if (len(set(individual)) != len(individual)):
        return False
    else:
        return True

def evaluate(individuals):
    """ 
    Evaluate function calculates the fitness values for each individual. 
    """
    value=[]
    limit_orders = 1000
    value.append(cost_distance.values[0,individuals[0]+1])                      # Save initial distance from the warehouse to the first Customer in the indivdual.
    
    for i in range(n_customers-1):
        limit_orders = limit_orders - orders['Orders'][individuals[i]+1]        # Keep track of how many orders it has on the truck
        if orders['Orders'][individuals[i+1]+1] > limit_orders :
            value.append(cost_distance.values[individuals[i]+1,0])              # If the  order of the next client is bigger than the number of the orders in the truck   
            value.append(cost_distance.values[0,individuals[i+1]+1])            # we save the current customer distance to warehouse and the distance from the warehouse to the next costumer.
            limit_orders = 1000                                                 # When he goes to the warehouse the salesman fills the truck with 1000 orders.

        value.append(cost_distance.values[individuals[i]+1,individuals[i+1]+1]) # Save the distance between the current costumer to the next.

    value.append(cost_distance.values[individuals[n_customers-1]+1,0])          # Save the distance from the last costumer to the warehouse.
    fit_ind = (sum(value))                                                              

    return fit_ind,

def heuristic(ware_location):
    max_x = 100
    heu_individual = []

    if ware_location == 'center': # The file of order has got a different head
        x = 'X'
        y = 'Y'
    else:
        x = 'x'
        y = 'y'

    location_aux = location[1:n_customers+1]                                       
    x_left = location_aux[location_aux[x] <= max_x/2]                               # Getting the customers only located in the left side of the map
    heu_individual_left =x_left.sort_values(by=[y])                                 #Sort those customers by lower to higher in the vertical axis only on for the costumers in the left side
    x_right = location_aux[location_aux[x] > max_x/2]                               # Getting the customers only located in the right side of the map
    heu_individual_right = x_right.sort_values(by=[y], ascending=False)             #Sort those customers by higher to lower in the vertical axis only on for the costumers in the left side
    heu_individual = heu_individual_left['Customer XY'].append( heu_individual_right['Customer XY']).tolist()  #Join both sorted vectors to create the individual

    return  [x-1 for x in heu_individual]

def minimization(iterations, number_cities, wareh, t_order, heu ):

    mean = []
    min_run =[]
    n_pop=100
    previous_bestfit=10000

    #Different probabilities for each case gives us better results. Thats why we choose the probabilities taking into account the case

    if number_cities == 10 and wareh == 'center' and t_order == 'file':
        CXPB , MUTPB = 0.8, 0.4
        name = 'WHCentral_OrdFile'
    if number_cities == 10 and wareh == 'corner' and t_order == 'file':
        CXPB , MUTPB = 0.6, 0.4
        name = 'WHCorner_OrdFile'
    if number_cities == 10 and wareh == 'center' and t_order == 'fixed':
        CXPB , MUTPB = 0.4, 0.6
        name = 'WHCentral_Ord50'
    if number_cities == 10 and wareh == 'corner' and t_order == 'fixed':
        CXPB , MUTPB = 0.6, 0.3
        name = 'WHCorner_Ord50'

    if number_cities == 30 and wareh == 'center' and t_order == 'file':
        CXPB , MUTPB = 0.5, 0.7
        name = 'WHCentral_OrdFile'
    if number_cities == 30 and wareh == 'corner' and t_order == 'file':
        CXPB , MUTPB = 0.6, 0.6
        name = 'WHCorner_OrdFile'
    if number_cities == 30 and wareh == 'center' and t_order == 'fixed':
        CXPB , MUTPB = 0.4, 0.8
        name = 'WHCentral_Ord50'
    if number_cities == 30 and wareh == 'corner' and t_order == 'fixed':
        CXPB , MUTPB = 0.6, 0.8
        name = 'WHCorner_Ord50'

    if number_cities == 50 and wareh == 'center' and t_order == 'file':
        CXPB , MUTPB = 0.6, 0.6
        name = 'WHCentral_OrdFile'
    if number_cities == 50 and wareh == 'corner' and t_order == 'file':
        CXPB , MUTPB = 0.5, 0.7
        name = 'WHCorner_OrdFile'
    if number_cities == 50 and wareh == 'center' and t_order == 'fixed':
        CXPB , MUTPB = 0.6, 0.8
        name = 'WHCentral_Ord50'
    if number_cities == 50 and wareh == 'corner' and t_order == 'fixed':
        CXPB , MUTPB = 0.5, 0.7
        name = 'WHCorner_Ord50'

    for iter in range(iterations):

        random.seed(34+iter)
        minimum= []
        g=0

        pop = toolbox.population(n_pop)                     #Creating the population

        if heu == 'yes':                                    #If we want to use the heuristic 
            pop[0]= toolbox.individual_heuristic()

        fitnesses = list(map(toolbox.evaluate, pop))        #Evaluate the initial population
        for ind, fit in zip(pop, fitnesses):                #Give a fitness to each individual
            ind.fitness.values = fit
        
        #GENERATIONS

        while g < 100:
            
            g = g + 1

            #SELECTION 

            offspring = toolbox.select(pop, len(pop))           #Selecting the offsprings 
            offspring = list(map(toolbox.clone, offspring))     

            #CROSSOVER

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

            # Getting the new population from the offsprings

            pop[:] = offspring
            
            #Getting the individual most minimized

            best_ind = tools.selBest(pop, 1)[0]
            best_ind1 = [x+1 for x in best_ind]

            #Saving the best individual of each generation to present the convergence curve

            minimum.append(best_ind.fitness.values[0])
        
        #Getting the best run

        if (best_ind.fitness.values[0] < previous_bestfit ):        
            min_run = minimum
            best_individual = best_ind1
            best_fit = best_ind.fitness.values[0]
        previous_bestfit = best_ind.fitness.values[0]

        #Getting the mean of the minimum of a run
        mean.append(best_ind.fitness.values[0])

    #Getting the mean and stander deviation and mean from the minimums of all the runs
    print('\nMean of the 30 runs: ',sum(mean)/len(mean))
    print('Standar deviation of the 30 runs: ',np.std(mean))
    print('The individual of the best run ', best_individual )
    print('Fitness of the best run', best_fit)

    return min_run, name


##################################  MAIN  ############################################

#Getting the parameters that we disire 
n_customers, warehouse, type_order, heuris, tourn_number,cost_distance, location, orders = get_parameters()

################################  MINIMIZATION CLASS  ###################################

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

###############################  POPULATION  ######################################

toolbox = base.Toolbox()

toolbox.register("individual_heuristic", tools.initIterate, creator.Individual, partial(heuristic,warehouse))
toolbox.register("individual", tools.initIterate, creator.Individual, partial(random.sample, range(0,n_customers), n_customers))
toolbox.register("population", tools.initRepeat, list, toolbox.individual) 

############################## OPERATORS  #####################################################

toolbox.register("evaluate", evaluate)
toolbox.decorate("evaluate", tools.DeltaPenalty(check_feasiblity, 10000)) 
toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)  
toolbox.register("select", tools.selTournament, tournsize=tourn_number)

###############################  MINIMIZATION  ##################################

runs = 30
best_min_run, name_case = minimization(runs, n_customers,warehouse, type_order, heuris)


plt.plot(best_min_run, label = name_case)
plt.xlabel('#Generations')
plt.ylabel('Total Distance for the best run')
plt.legend(loc='upper right')
plt.title('Convergence Curve for %s customers', n_customers)
plt.show()

