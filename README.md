# Smoothed-Particle Hydrodynamics Simulator
<hr>
The SPH Framework encompasses the command line environment and configuration file interpretation. One Python script, and three Python modules are included to accomplish this: SPH.py, CLI.py, Configuration.py, Framework.py, and Particle.py. 

### Usage
The SPH simulator needs either a CSV file as input, or it needs to be instructed to generate a set amount of particles. Most options have defaults pre-set in the config file
(generally <b>sph.conf</b>).
<br><br>Some simple examples are shown below:
* If all you want to do is generate 100 particles, you would use: ```python sph.py -g 100```
* Reading from an input file is as simple as: 
    ```python sph.py -i example.csv```
* Having a custom output file prefix can be useful too: 
    ```python sph.py -i example.csv -s output```
  * This would yield output files such as: <b>output-1.csv</b>, <b>output-100.csv</b>, <b>output-200.csv</b>, etc.

#### Available Flags
```
sph.py [-h] [-g GEN] [-i IFILE] [--gtype {gaussian,random}]
          [-s SAVEFILE] [--bound BOUND] [--stdev STDEV]
          [--maxiter MAXITER] [--t_norm {months,years,decades,centuries}]
          [--x_norm {Meters,kilometers,light-years}]
          [--kernel {gaussian,cubic}]
          [--mode {serial,parallel}]

```

### SPH.py
The entrypoint for the SPH simulator application.
* Aggregates the Configuration and CLI Python modules
* Provides a chokepoint for catching exceptions

### Configuration.py
Responsible for interpreting the simulator's configuration file. The configuration file is used to set default values in the simulation when they haven't been explicitly set in the initial program invocation.
* Parses configuration file
* Creates new configuration file if missing
* Returns configuration files to other SPH modules

### CLI.py
Builds the command line interface, and is the only interaction between the user and the simulation.
* Interprets user arguments, and attempts to resolve errors in program input
* Generates particles if '-g | --gen' option is specified
* Reads in particles from an input file if '-i | --input' option is specified
* Periodically save program state by writing particles to output files

### Framework.py
Calculates particle accelerations and numerically integrates their equations of motion

### Particle.py
Simply stores particle IDs, X, Y, and Z coordinates, mass, velocity, and acceleration for each particle.

### SPH.conf
This is the configuration file for the simulator. Options take a 'key=value' format. Newlines are ignored, and lines are considered commented when #'s are found at the beginning of them. In addition, each option has a small description above it in the .conf file.

### Generating 3D Plots
Command line usage:
```
3dplot.py [-h] [--prefix PREFIX] [--fps FPS] [--file FILE]
                 [--title TITLE]
```
Basic example: ```python 3dplot.py --prefix output```

Advanced example: ```python 3dplot.py --prefix output --fps=15 --file test.gif --title="multiword title use quotes"```

The prefix parameter is the prefix of the output files that contain particle locations at a given timestep (<b>output</b>-1.csv, <b>output</b>-100.csv, <b>output</b>-200.csv, etc).

### Necessary Dependencies
* Python 2.7 - Tested with Python 2.7.8 and 2.7.9
* Numpy - Used for the Particle.py module to build arrays
* MatPlotLib - Needed to generate 3d plots by <b>3dplot.py</b>
