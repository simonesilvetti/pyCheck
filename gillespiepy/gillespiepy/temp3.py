from numba import guvectorize, float64, int64


@guvectorize([(float64[:], float64[:], float64[:])], '(n),(n)->(n)')
def calculate_rates(coeff, species, res):
    res[0]=coeff[0]/100*species[0]*species[1]
    res[1]=coeff[1]*species[1]