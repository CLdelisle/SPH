# SPH Input/Output
<hr>
The I/O Framework encompasses the command line environment and configuration file interpretation. One Python script, and three Python modules are included to accomplish this: SPH.py, CLI.py, Configuration.py, and Particle.py. 

### Usage
The SPH simulator needs either a CSV file as input, or it needs to be instructed to generate a set amount of particles. Most options have defaults pre-set in the config file
(generally <b>sph.conf</b>).
<br><br>Some simple examples are shown below:
<ul>
    <li>If all you want to do is generate 100 particles, you would use: ```python sph.py -g 100```</li>
    <li>Reading from an input file is as simple as: ```python sph.py -i example.csv```</li>
    <li>Having a custom output file prefix can be useful too: ```python sph.py -i example.csv -s output```
    <ul><li>This would yield output files such as: <b>output-001.csv</b>, <b>output-100.csv</b>, etc.</li></ul>
</ul>

#### Available Flags
```
sph.py [-h] [-g GEN] [-i IFILE] [--gtype {gaussian,random}]
          [-s SAVEFILE] [--bound BOUND] [--stdev STDEV]
          [--maxiter MAXITER] [--t_norm {months,years,decades,centuries}]
          [--x_norm {Meters,kilometers,light-years}]
          [--kernel {gaussian,cubic}]
```

### SPH.py
The entrypoint for the SPH simulator application.
<ul>
    <li>Aggregates the Configuration and CLI Python modules</li>
    <li>Provides a chokepoint for catching exceptions</li>
</ul>

### Configuration.py
Responsible for interpreting the simulator's configuration file. The configuration file is used to set default values in the simulation when they haven't been explicitly set in the initial program invocation. 
<ul>
    <li>Parses configuration file</li>
    <li>Creates new configuration file if missing</li>
    <li>Returns configuration files to other SPH modules</li>
</ul>

### CLI.py
Builds the command line interface, and is the only interaction between the user and the simulation.
<ul>
    <li>Interprets user arguments, and attempts to resolve errors in program input</li>
    <li>Generates particles if '-g | --gen' option is specified</li>
    <li>Reads in particles from an input file if '-i | --input' option is specified</li>
    <li>Periodically save program state by writing particles to output files</li>
</ul>

### Particle.py
Simply stores particle IDs, X, Y, and Z coordinates, mass, velocity, and acceleration for each particle.

### SPH.conf
This is the configuration file for the simulator. Options take a 'key=value' format. Newlines are ignored, and lines are considered commented when #'s are found at the beginning of them. In addition, each option has a small description above it in the .conf file.

### Necessary Dependencies
<ul>
    <li>Python 2.7 - Tested with Python 2.7.8 and 2.7.9</li>
    <li>Numpy - Used for the Particle.py module to build arrays</li>
</ul>
