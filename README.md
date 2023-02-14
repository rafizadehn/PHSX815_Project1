# Project 1 - Velocities of Gas Particles in a Boltzmann Distribution

**Formal write up is located in this repository, named "Rafizadeh_WRITEUP.pdf"**

## Data Generation

The velocity measurements are made by the `rng_MaxBoltz.py` python file. This file requires python3 to run, and includes the following packages listed at the top of the script:

```
  import sys
  import numpy as np
  import matplotlib.pyplot as plt
  from scipy.special import erf
  from scipy.interpolate import interp1d as interp
```

To run this script from the terminal in linux, run:

> $ python3 rng_MaxBoltz.py

This runs the file with the default parameters, which are: 275 K as the first temperature, 320 K as the second temperature, 1000 as the default number of particles, 85 amu as the default molecular mass, and 5555 as the default seed for the random number generator.

These values can be altered from the command line in the terminal by simply adding an argument after the file name. The arguments to change these include `-temp1`, `-temp2`, `-Nmeas`, `-mass`, `-seed` for those values, respectively. 

For example, it may looks something like this in linux:

> $ python3 rng.MaxBoltz.py -temp1 100 -temp2 1000

which would generate the values at 100 K and 1000 K, and keep the other parameters the same. The parameters chosen for this file are written to a text file called `parameters.txt`. This allows for the analysis script to read the parameters that were used in the generation file for more efficient analysis.

## Data Analysis

The generated data is plotted and analyzed by the `plot_MaxBoltz.py` python file. This file requires python3 to run, and includes the following packages listed at the top of the script:

```
  import sys
  import numpy as np
  import matplotlib.pyplot as plt
  import scipy.stats as ss
  from scipy.stats import ttest_ind
  import math
  from statsmodels.stats.power import  tt_ind_solve_power
```

To run this script from the terminal in linux, run:

> plot_MaxBoltz.py

This creates plot from an input data set and analyzes it. By default, it will analyze the data generated and outputted by the rng_MaxBoltz.py file, yet any data file can be used as an input by use of some arguments.

### Using the `rng_MaxBoltz.py` generated data:

The anaylsis file is set to read the output of the generator file by default, no arguments need to be added for the data import. The only arguments that you may want to change is the number of bins plotted in the histogram graph, and the type I error given for the statistical power analysis. These can be defined with `-nbins` and `-alpha` by,

> $ python3 plot_MaxBoltz.py -nbins 50 -alpha 0.05

which sets the number of bins used in the histogram plot as 50 bins, and the type I error allowance as 5%.

The script will then plot the data as a histogram, and provide a stastical power analysis for different values of Nmeas, the number of particles measured.

### NOT using the `rng_MaxBoltz.py` generated data:

Any set of similarly formatted data can be put into this file to be run through the use of the `-temp1`, `-temp2`, and `-param` arguments. The `-temp1` argument specifies the name of the .txt file containing the velocity data at the first temperature value. Similarly, `-temp2` specifies the name of the .txt file that contains the velocity data at the second temperature value. The `-param` argument specifies the name of the .txt file containing the values of the temperature of the gases, the molecular mass, and the number of particles generated.

![pic1](https://user-images.githubusercontent.com/76142511/218628962-a6cbf1ad-71e7-4460-99df-b18df8cb4931.png)

![pic2](https://user-images.githubusercontent.com/76142511/218628977-a798c0cc-4cb5-47c1-90d6-a6e9817c5a0e.png)


