When testing SOOP.py the user will first be prompted to input:
 -The number of cities for which he wishes to test the EA:  10, 30 or 50 
 -The location of the warehouse for which he wishes to test the EA: center or corner 
 -The number of orders for which he wishes to test the EA: file or fixed  (file to use the orders which are in the file and fixed to have all the orders equal to 50)
-The use of the heuristic: yes or no (yes if we want the results using the heuristic or no if we want the results without the heuristic)
After that, EA will start to run for 30 iterations, each one with a different random seed. After running the 30 iterations it will be printed on the console
some statistics which represent:
-Mean of the minimum of the 30 runs
-The Stander deviation of the minimums of the 30 runs
-The individual of the best run
-A plot of #Generations in the horizontal axis and the total distance for the best run in the vertical axis


When testing MOOP.py the user will first be prompted to input the number of cities for which he wishes to test the EA, here the available options are 10, 30 or 50 cities.
After that, EA will start to run for 30 iterations, each one with a different random seed, where in each iteration a sequence of logbook registers of the statistics of the 
population of each generation will be printed on the console. This statistics include number of evaluations per generation, the respective standard deviation, average, 
min and max values of fitnesses of each population.
Lastly, it will show on the console the values of distance and travel cost when the minimum cost and minimum distance is achieved, respectively.
The user also as the option to uncomment some of the lines if he wishes to see some plots relative to the evolution of the populations.  

