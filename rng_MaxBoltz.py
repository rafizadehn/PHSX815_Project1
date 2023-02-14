import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.interpolate import interp1d as interp

# import our Random class from Random.py file
sys.path.append(".")
from Random import Random

def boltz(v,m,T): # defines a function that returns a PROBABILITY for a specific set of parameters v, m, and T
    kB = 1.38e-23
    return (m/(2*np.pi*kB*T))**1.5 * 4*np.pi * v**2 * np.exp(-m*v**2/(2*kB*T))

def inv_boltz(v,m,T): # inverse function inverts it for ease of distribution calculation
    kB = 1.38e-23
    a = np.sqrt(kB*T/m)
    return erf(v/(np.sqrt(2)*a)) - np.sqrt(2/np.pi)* v* np.exp(-v**2/(2*a**2))/a

def velocities(n, seed):
    # this uses a random number generator from 
    # Dr. Christopher Rogan's GitHub repository.
    # If you cannot access this, numpy can be used with:
    # np.random.rand(n)
    random = Random(seed)
    vals = []
    for i in range(0, n):
        vals.append(random.rand())
    vel = inv_cdf(vals)
    return vel

if __name__ == "__main__":
    if '-h' in sys.argv or '--help' in sys.argv:
        print("Usage: %s -temp1 [temperature (k)] -temp2 [temperature (k)] -seed [seed] -Nmeas [number of particles sampled] -mass [mass in amu]" % sys.argv[0])
        print
        sys.exit(1)

    # default temp1 (in kelvin)
    T1 = 275

    # default temp2 (in kelvin)
    T2 = 320
    
    # default number of particles sampled (Nmeas) 
    N = 1000

    # default seed
    seed = 5555

    # default mass (in amu)
    m = 85 
    
    # read the user-provided inputs from the command line (if there)
    if '-seed' in sys.argv:
        p = sys.argv.index('-seed')
        seed = sys.argv[p+1]
    if '-Nmeas' in sys.argv:
        p = sys.argv.index('-Nmeas')
        N = int(sys.argv[p+1])
    if '-mass' in sys.argv:
        p = sys.argv.index('-mass')
        m = int(sys.argv[p+1])
    if '-temp1' in sys.argv:
        p = sys.argv.index('-temp1')
        T1 = int(sys.argv[p+1])
    if '-temp2' in sys.argv:
        p = sys.argv.index('-temp2')
        T2 = int(sys.argv[p+1])
    
    # exports inputs to a parameter file to be easily transfered for analysis in plot_MaxBoltz.py
    parameters = [seed, N, m, T1, T2] 
    with open(r'parameters.txt', 'w') as fp:
        for item in parameters:
            fp.write("%s\n" % item)

    # constants
    amu = 1.66e-27 # conversion factor for mass to SI
    mass = m * amu # converts input to kg
    v = np.arange(0,800,1) # creates x-axis values for plot
    vs = np.arange(0,2500,0.1) # creates values to input into density function equation, needs different range from the plot values due to the inversion of the Boltzmann equation.

    ###### TEMP1:

    cdf = inv_boltz(vs,mass,T1) # evaluates the density function equation at the given input values, returns the parameterized function that can then be interpolated
    inv_cdf = interp(cdf,vs,fill_value="extrapolate") # interpolation of previously parameterized equation
    vel = velocities(N, seed)

    with open(r'temp1.txt', 'w') as fp: # writes data to external text file
        for item in vel:
            fp.write("%s\n" % item)

    ####### TEMP 2:

    cdf2 = inv_boltz(vs, mass, T2) # same as temp1
    inv_cdf = interp(cdf2, vs, fill_value="extrapolate")
    vel1 = velocities(N, seed)

    with open(r'temp2.txt', 'w') as fp:
        for item in vel1:
            fp.write("%s\n" % item)


