import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.interpolate import interp1d as interp

# import our Random class from Random.py file
sys.path.append(".")
from Random import Random

def boltz(v,m,T):
    kB = 1.38e-23
    return (m/(2*np.pi*kB*T))**1.5 * 4*np.pi * v**2 * np.exp(-m*v**2/(2*kB*T))

if __name__ == "__main__":
    if '-h' in sys.argv or '--help' in sys.argv:
        print("Usage: %s -temp1 [temp1 file] -temp2 [temp2 file] -param [parameters file] -nbins [number of bins]" % sys.argv[0])
        print
        sys.exit(1)
 
    # default temp1 file name
    temp1 = 'temp1.txt'

    # default temp2 file name
    temp2 = 'temp2.txt'

    # default parameters file name
    param = 'parameters.txt'
    
    # default number of bins
    nb = 40

    # read the user-provided inputs from the command line (if there)
    if '-temp1' in sys.argv:
        p = sys.argv.index('-temp1')
        temp1 = sys.argv[p+1]
    if '-temp2' in sys.argv:
        p = sys.argv.index('-temp2')
        temp2 = sys.argv[p+1]
    if '-param' in sys.argv:
        p = sys.argv.index('-param')
        m = sys.argv[p+1]
    if '-nbins' in sys.argv:
        p = sys.argv.index('-nbins')
        nb = int(sys.argv[p+1])

    ## import velocities of particles from gas at T1
    vel1 = []
    with open(temp1) as fp:
        for line in fp:
            line=float(line)
            vel1.append(line)
    
    ## import velocities of particles from gas at T2
    vel2 = []
    with open(temp2) as fp:
        for line in fp:
            line=float(line)
            vel2.append(line)
    
    ## import parameters used from rng file
    parameters = []
    with open(param) as fp:
        for line in fp:
            line=float(line)
            parameters.append(line)

    ## extrapolate parameters from param file
    seed, N, m, T1, T2 = parameters

    # constants
    amu = 1.66e-27 # conversion factor for mass to SI
    mass = m * amu # converts input to kg
    v = np.arange(0,800,1) # creates x-axis values for plot
    vs = np.arange(0,2500,0.1) # creates values to input into density function equation, needs different range from the plot values due to the inversion of the Boltzmann equation.

    fig = plt.figure() # creates plot environment for all plots
    ax = fig.add_subplot() # allows for subplots, two different histograms will be plotted on the same plot

    ax.set_xlabel('Speed (m/s)', fontsize = 15)
    ax.set_ylabel('Probability Density', fontsize = 15)
    ax.set_title('Velocities of Particles in a Gas with Molecular Mass m = '+str(int(m))+' amu', fontsize = 15)
    ax.tick_params(axis='both', labelsize=13)

    # plot the histograms of data
    
    ax.hist(vel1,bins=nb,density=True,fc='salmon',alpha=0.4,lw=0.6, label='T = '+str(int(T1))+' K', edgecolor = 'k')
    ax.hist(vel2,bins=nb,density=True,fc='c',alpha=0.4,lw=0.6, label='T = '+str(int(T2))+' K', edgecolor = 'k')

    # graph the actual calcualted Boltzmann ditribution for given input values
    vs = np.arange(0,1500)
    fv = boltz(vs,mass,T1)
    ax.plot(vs,fv,'salmon',lw=2)

    vs = np.arange(0,1500)
    fv = boltz(vs,mass,T2)
    ax.plot(vs,fv,'c',lw=2)

    plt.legend(fontsize = 15)
    plt.show()




