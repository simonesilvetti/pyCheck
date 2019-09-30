import numpy as np
from typing import Callable


def direct_method(end_time: int,
                  end_steps: int,
                  rate_function: Callable[[np.ndarray, np.ndarray], np.ndarray],
                  rates: np.ndarray,
                  species: np.ndarray,
                  reaction_effects: np.ndarray):
    """
    The direct simulation method introduced in Gillespie 1977.
    :param end_time:
    :param end_steps:
    :param rate_function:
    :param rates:
    :param species:
    :param reaction_effects:
    :return:
    """
    simulation_time = 0
    time_step = 0
    random_count = 0

    # TODO: Profile this to establish trade off between function calls and cache misses.
    log_random_values = -np.log(np.random.random(1000))
    random_values = np.random.random(1000)

    # TODO: Convert steps vs time to closures.
    while simulation_time < end_time and time_step < end_steps:

        rates = rate_function(rates, species)
        total_rate = rates.sum()

        if total_rate <= 0:
            break

        if random_count == 1000:
            log_random_values = -np.log(np.random.random(1000))
            random_values = np.random.random(1000)
            random_count = 0

        random_cutoff = random_values[random_count]
        time_to_next_reaction = log_random_values[random_count] / total_rate
        simulation_time += time_to_next_reaction

        true_cutoff = random_cutoff * total_rate
        # TODO: Compare cumsum to just normalizing
        cutoffs = np.cumsum(rates)
        selected_reaction = np.searchsorted(cutoffs, true_cutoff, side='right')
        species += reaction_effects[selected_reaction]
        time_step += 1
        random_count += 1
    return species, rates,simulation_time

def direct_method_parallel(end_time: int,
                  end_steps: int,
                  rate_function: Callable[[np.ndarray, np.ndarray], np.ndarray],
                  coeff: np.ndarray,
                  species: np.ndarray,
                  reaction_effects: np.ndarray,
                    runs:int):
    """
    The direct simulation method introduced in Gillespie 1977.
    :param end_time:
    :param end_steps:
    :param rate_function:
    :param rates:
    :param species:
    :param reaction_effects:
    :return:
    """
    simulation_time = np.zeros(runs)
    time_step = 0
    random_count = 0

    # TODO: Profile this to establish trade off between function calls and cache misses.
    log_random_values = -np.log(np.random.rand(1000,runs))
    random_values = np.random.rand(1000,runs)
    totaltime=np.array([])

    # TODO: Convert steps vs time to closures.
    #while min(simulation_time) < end_time and time_step < end_steps:
    #fun = lambda x: rate_function(coeff[0], x)
    while min(simulation_time) < end_time:
        #rate_function_parallel = lambda x : rate_function(rates,x)
        #rates = np.apply_along_axis(rate_function_parallel, 1, species)


        #rates = np.array([rate_function(a, b) for a, b in zip(coeff, species)])
        #rates=np.array([rate_function(coeff[0], x) for x in species])
        rates=rate_function(coeff,species[:,0:2])
        #rates = np.apply_along_axis(fun, 1, species)
        #rates = np.apply_along_axis(fun,1,species)
        # rates = rate_function(rates, species)
        total_rate = rates.sum(1)

        if min(total_rate) <= 0:
            #i = np.argmin(total_rate)
            i=np.where(total_rate<=0)
            rates=np.delete(rates, i,axis=0)
            species=np.delete(species, i,axis=0)
            total_rate = np.delete(total_rate, i)
            totaltime = np.append(totaltime, simulation_time[i])
            simulation_time=np.delete(simulation_time, i)
            coeff=np.delete(coeff, i,axis=0)
            if(len(species)==0):
                break


        if random_count ==len(species):
            log_random_values = -np.log(np.random.rand(1000,len(species)))
            random_values = np.random.rand(1000,len(species))
            random_count = 0

        random_cutoff = random_values[random_count][0:len(species)]
       # time_to_next_reaction = log_random_values[random_count] / total_rate
        time_to_next_reaction=np.array([a / b if b != 0 else 0 for a, b in zip(log_random_values[random_count][0:len(species)], total_rate)])
        simulation_time += time_to_next_reaction

        true_cutoff = random_cutoff * total_rate
        # TODO: Compare cumsum to just normalizing
        cutoffs = np.cumsum(rates,1)
        #selected_reaction =searchsortedNew(cutoffs,true_cutoff)
        selected_reaction = np.array([np.searchsorted(a, b, side='right') if b!=0 else 0 for a,b in zip(cutoffs,true_cutoff)])
        # if(max(selected_reaction)==2):
        #     print('ao')
        species += reaction_effects[selected_reaction]
        time_step += 1
        random_count += 1
    return species, rates,totaltime


algorithms_mapping = {
    "direct": direct_method,
    "direct_parallel":direct_method_parallel
}