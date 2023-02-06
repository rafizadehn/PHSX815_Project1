import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.interpolate import interp1d as interp
import scipy.stats as ss
from scipy.stats import ttest_ind
import math
from statsmodels.stats.power import  tt_ind_solve_power

# import our Random class from Random.py file
sys.path.append(".")
from Random import Random

# import our MySort class from MySort.py file
from MySort import MySort

def boltz(v,m,T):
    kB = 1.38e-23
    return (m/(2*np.pi*kB*T))**1.5 * 4*np.pi * v**2 * np.exp(-m*v**2/(2*kB*T))

def PSD(s1,s2):
    n1, n2 = len(s1), len(s2)
    var1, var2 = np.var(s1, ddof=1), np.var(s2, ddof=1)
    num = ((n1-1) * var1) + ((n2-1) * var2)
    den = n1+n2-2
    return np.sqrt(num/den)

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
    # ax.set_title('Velocities of Particles in a Gas with Molecular Mass m = '+str(int(m))+' amu', fontsize = 15)
    ax.tick_params(axis='both', labelsize=13)

    # plot the histograms of data
    
    ax.hist(vel1,bins=nb,density=True,fc='salmon',alpha=0.4,lw=0.6, label='T = '+str(int(T1))+' K', edgecolor = 'k')
    ax.hist(vel2,bins=nb,density=True,fc='c',alpha=0.4,lw=0.6, label='T = '+str(int(T2))+' K', edgecolor = 'k')

    # graph the actual calcualted Boltzmann ditribution for given input values
    vs = np.arange(0,1500)
    fv = boltz(vs,mass,T1)
    ax.plot(vs,fv,'salmon',lw=2)
    
    plt.grid(True, alpha = 0.7, linestyle='--')

    vs = np.arange(0,1500)
    fv = boltz(vs,mass,T2)
    ax.plot(vs,fv,'c',lw=2, linestyle = '--')

    plt.legend(fontsize = 15)

    ## Analysis

    # calculates t-statistic and p-value between both distributions
    tstat, pv = ttest_ind(vel1, vel2)
    
    mean1 = np.mean(vel1)
    mean2 = np.mean(vel2)

    std1 = np.std(vel1)
    std2 = np.std(vel2)
   
    # cumulative distribution function 

    # ax.plot(vs, ss.norm.cdf(vs, mean2, std2), label='pdf')
    # ax.plot(vs, ss.norm.cdf(vs, mean1, std1), label='cdf')

    cdf1 = ss.norm.cdf(vs, mean1, std1)
    cdf2 = ss.norm.cdf(vs, mean2, std2)

    # sorter

    Sorter = MySort()
    s_vel1 = Sorter.QuickSort(vel1)
    s_vel2 = Sorter.QuickSort(vel2)

    crit_l1 = s_vel1[int(0.95 * N)]
    crit_l2 = s_vel2[int(0.95 * N)]

    s_pooled = PSD(vel1, vel2)
    cd = (mean2-mean1)/s_pooled

    effect_size = cd
    sample_size = N
    alpha = .05
    ratio = 1.0
    power = []
    percentiles = np.arange(0.005, 1, 0.01)

    for i in percentiles:
        vel1_temp = vel1[0:int(i*N)]
        vel2_temp = vel2[0:int(i*N)]
        sp_temp = PSD(vel1_temp, vel2_temp)
        cd_temp = (np.mean(vel2_temp)-np.mean(vel1_temp))/sp_temp
        statistical_power = tt_ind_solve_power(effect_size=cd_temp, nobs1=len(vel1_temp), alpha=alpha, ratio=1.0, alternative='two-sided')
        power.append(statistical_power)
 
    plt.show()
   
    plt.figure()
    plt.scatter(percentiles*N, power, c = 'crimson', alpha = 0.7, label= 'T = '+str(T2)+' K')
    plt.xlim([0, N])
    plt.ylim([0.05, 1])
    plt.xlabel('Number of Particles Sampled', fontsize = 15)
    plt.ylabel(r'Power (1 - $\beta$)', fontsize = 15)
    plt.tick_params(axis='both', labelsize = 13)
    plt.title('Statistical Power of H1 Compared to H0 (275 K) Per Sample Size', fontsize = 15, fontweight = "bold")
    plt.legend(loc='lower right', fontsize = 15)
    plt.show()

    with open(r'power_values.txt', 'w') as fp:
        for item in power:
            fp.write("%s\n" % item)
    


