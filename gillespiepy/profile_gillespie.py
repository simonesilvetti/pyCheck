import cProfile
import numpy as np
import timeit
import pstats

import gillespiepy.gillespiepy.PyscesParser as parser
from gillespiepy.gillespiepy.Constants import MODEL_DIRECTORY

STATS_FILE = "stats"
SYSTEM = "epidemic.psc"
#SIMPLE_SYSTEM = "BirthDeath.psc"


def run_simulation(rates,runs,timeFinal):
    reaction_system = parser.parse(SYSTEM, MODEL_DIRECTORY)
    species, rates, endTime = reaction_system.run_simulation(rates,runs,timeFinal)
    #print(species)
    #print(endTime)
    #print(np.sum(endTime>=100 and endTime<=120 ))
    return np.sum(np.array([x >= 100 and x <= 120 for x in endTime])) / len(endTime)


#a = timeit.Timer(lambda:run_simulation())

#print(a.timeit(number=10))
#a=run_simulation()
#a=cProfile.run('run_simulation()', STATS_FILE)
#print(a)
# stats = pstats.Stats(STATS_FILE)
# stats.strip_dirs()
# stats.sort_stats('tottime')
# stats.print_stats(5)
