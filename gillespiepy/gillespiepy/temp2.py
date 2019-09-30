from numba import jit
@jit(nopython=True)
def calculate_rates(coeff, species):
  #rates[0] = coeff[0]/100*species[0]*species[1]
  #rates[1] = coeff[1]*species[1]
  return [coeff[0]/100*species[0]*species[1],coeff[1]*species[1]]
