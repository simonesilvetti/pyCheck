import numpy as np
from numba import guvectorize, float64, int32

from pycheck.regressor.supportFunction import searchsortedNew, divide


def direct_method_parallel(end_time: int,
                           coeff: np.ndarray,
                           species: np.ndarray,
                           reaction_effects: np.ndarray,
                           runs: int):
    size = coeff.shape[0] * runs
    simulation_time = np.zeros(size)
    time_step = 0
    random_count = 0
    species = np.tile(species.copy(), (size, 1))
    res = np.tile(coeff[0, :], (runs, 1))
    for i in range(1, coeff.shape[0]):
        res = np.vstack((res, np.tile(coeff[i, :], (runs, 1))))
    coeff = res
    log_random_values = -np.log(np.random.rand(1500, len(species)))
    random_values = np.random.rand(1500, len(species))
    totaltime = np.array([])
    while min(simulation_time) < end_time:
        species_ = species[:, 0:2]
        rates = calculate_rates(coeff, np.int32(species_))
        total_rate = rates.sum(1)
        if min(total_rate) <= 0:
            i = np.where(total_rate <= 0)
            rates = np.delete(rates, i, axis=0)
            species = np.delete(species, i, axis=0)
            total_rate = np.delete(total_rate, i)
            totaltime = np.append(totaltime, simulation_time[i])
            simulation_time = np.delete(simulation_time, i)
            coeff = np.delete(coeff, i, axis=0)
            if (len(species) == 0):
                break

        if random_count == len(species):
            log_random_values = -np.log(np.random.rand(1500, len(species)))
            random_values = np.random.rand(1500, len(species))
            random_count = 0

        random_cutoff = random_values[random_count][0:len(species)]
        time_to_next_reaction = divide(log_random_values[random_count][0:len(species)], total_rate)
        simulation_time += time_to_next_reaction

        true_cutoff = random_cutoff * total_rate
        # TODO: Compare cumsum to just normalizing
        cutoffs = np.cumsum(rates, 1)
        selected_reaction = searchsortedNew(cutoffs[:, 0], true_cutoff)
        species += reaction_effects[selected_reaction]
        time_step += 1
        random_count += 1
    return species, totaltime


@guvectorize([(float64[:], int32[:], float64[:])], '(n),(n)->(n)')
def calculate_rates(coeff, species, res):
    res[0] = coeff[1] / 100 * species[0] * species[1]
    res[1] = coeff[0] * species[1]


def simulate(x, n):
    species, endTime = direct_method_parallel(125, np.array([x]), np.array([95, 5, 0]),
                                              np.array([[-1, 1, 0], [0, -1, 1]]), n)
    print(".", end="")
    return np.sum(np.array([100 <= x <= 120 for x in endTime])) / n
