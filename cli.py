import argparse
import sys
import configuration
import random
import os
from random import uniform, gauss


class Interface():

    def __init__(self):
        self.config = configuration.Config()
        defaults = self.config.getArg('all')
        # Set argument options for CLI
        self.parser = argparse.ArgumentParser(description="Run the SPH simulation")
        self.parser.add_argument("-g", "--gen", help="Number of particles to generate. Conflicts with [IFILE] argument",
                                type=int)
        self.parser.add_argument("-i", "--ifile", help="Input file path to read particles from. "
                                "Takes precedence over [GEN] argument")
        self.parser.add_argument("-o", "--ofile", help="Output file path to write particles to. Suffix is currently "+defaults['ofile'])
        self.parser.add_argument("--bound", help="Sets boundaries of particle space. Default is "+defaults['bound'],
                                type=int, default=int(defaults['bound']))
        self.parser.add_argument("--stdev", help="Standard deviation of particle space. Default is "+defaults['stdev'],
                                type=float, default=float(defaults['stdev']))
        self.parser.add_argument("--maxiter", help="Maximum iterations to run the simulation through. Default is "+defaults['maxiter'],
                                type=int, default=int(defaults['maxiter']))
        self.parser.add_argument("--t_norm", help="Time normalization. Default is "+defaults['t_norm'],
                                choices=['months', 'years', 'decades', 'centuries'], default=defaults['t_norm'])
        self.parser.add_argument("--x_norm", help="Space normalization. Default is "+defaults['x_norm'],
                                choices=['Meters', 'kilometers', 'light-years'], default=defaults['x_norm'])
        self.parser.add_argument("--kernel", help="Kernel function to use. Default is "+defaults['kernel'],
                                choices=['gaussian', 'random'], default=defaults['kernel'])
        self.parser.add_argument("--vidlen", help="Maximum length (in seconds) of video to produce", type=int)
        self.parser.add_argument("--res", help="Resolution of the simulation to use", default=defaults['res'])
        # Actually begin to parse the arguments
        self.args = self.parser.parse_args()

    ######################################################
    # Generate a specified number of particles in a 3D space using random.gauss() function
    # Should cluster particles in a sort of 'globe' in the center of the grid
    # INPUT: num (number of particles to generate)
    # OUTPUT: ppos (double array containing particle IDs and positions)
    ######################################################
    def createGaussian(self, num):
        mean = (self.args.bound/2)
        ppos = [] # array containing particle positions
        for i in range(0, num):
            particle = []
            particle.append(i) # assign particle id
            val = -1 # coordinate value for X, Y, or Z
            for j in range(1, 4):
                while val < 0 or self.args.bound < val: # ensure val is within range of particle boundaries
                    val = round(gauss(mean, self.args.stdev), 6)
                particle.append(val)
                val = -1  # reset val so the while loop will run again (otherwise X=Y=Z)
            ppos.append(particle)
        return ppos

    ######################################################
    # Generate a specified number of particles completely randomly
    # Given enough particles, the space should be about equally distributed
    # INPUT: num (number of particles to generate)
    # OUTPUT: ppos (double array containing particle IDs and positions)
    ######################################################
    def createRandom(self, num):
        bound = self.args.bound
        ppos = [] # array containing particle positions
        for i in range(0, num):
            # Particle created of the form: [PID, X, Y, Z]
            x = round(random.uniform(0, bound), 6)
            y = round(random.uniform(0, bound), 6)
            z = round(random.uniform(0, bound), 6)
            ppos.append([i, x, y, z])
        return ppos

    ######################################################
    # Will determine what kind of particle generation will occur
    # Mainly just a wrapper function for calling createRandom() or createGaussian()
    # INPUT: num (number of particles to generate), method (how to generate particles)
    # OUTPUT: ppos (double array containing particle IDs and positions)
    ######################################################
    def genParticles(self, num, method):
        if method == 'gaussian':
            print "[+] Generating particles according to Gaussian distribution"
            ppos = self.createGaussian(num)
        elif method == 'random':
            print "[+] Spreading particles randomly"
            ppos = self.createRandom(num)
        else:
            print "[+] No particle generation method selected. Spreading particles randomly"
            ppos = self.createRandom(num)
        
        # takes filename specified in sph.config, particle positions, and number of particles generated
        self.writeParticlesToFile(self.config.getArg('savefile'), ppos, num)
        return ppos

    ######################################################
    # Called when --ifile flag is set
    # Reads from an input file (orly?) of form "PID,X-coord,Y-coord,Z-coord" on each line
    # INPUT: none - grabs input filename from args parameter
    # OUTPUT: ppos (double array containing particle IDs and positions)
    ######################################################
    def readInputFile(self):
        file = self.args.ifile
        # particle positions - just like self.genParticles
        ppos = []
        with open(file, "r") as ifile:
           print "[+] Reading from input file \"%s\"" % file
           for line in ifile:
              # header = "Particle ID, X-coord, Y-coord, Z-coord\n"
              # strip '\n' from line, then split into a list at commas
              p = line.strip().split(",")
              # must cast values, because they are read in as strings by Python
              # casting should catch odd values as well (e.g. '40a' for a PID)
              ppos.append([int(p[0]), float(p[1]), float(p[2]), float(p[3])])
        return ppos

    ######################################################
    # This is where all the CLI rules will be set and interpreted, and further function calls done
    # Contradicting options are corrected here (e.g. --ifile and --gen flags set)
    # Does NOT do any direct data processing. Only calls other functions
    # INPUT: none
    # OUTPUT: particles (array of particles with form [PID, X-coord, Y-coord, Z-coord])
    ######################################################
    def setSimRules(self):
        if self.args.ifile: # If [IFILE] is specified, ignore everything else
            particles = self.readInputFile()
        elif not self.args.ifile and self.args.gen: # If [IFILE] isn't specified and [NUMPRT] is specified, generate particles
            particles = self.genParticles(self.args.gen, self.args.kernel)
        else: # If [IFILE] and [NUMPRT] are NOT specified, print help message and exit
            self.parser.print_help()
            print "\n[-] You did not specify an input file or tell me to generate particles!"
            sys.exit(1)
        # retrieved from either an input file or generated in self.genParticles
        return particles

    ######################################################
    # Writes particle positions [PID, X-coord, Y-coord, Z-coord] to a file, line-by-line
    # Should be primarily used to save particle positions during a simulation
    # Should be the first function called after self.genParticles()
    # INPUT: fname (filename to write to), ppos (particles list to write), num (number of particles)
    # OUTPUT: None directly, but an output file will be generated
    ######################################################
    def writeParticlesToFile(self, fname, ppos, num):
        with open(fname, "w") as output:
            for i in range(0, num):
                # header = "Particle ID, X-coord, Y-coord, Z-coord\n"
                if i < (num-1):
                    line = "%d,%f,%f,%f\n" % (int(ppos[i][0]), float(ppos[i][1]), float(ppos[i][2]), float(ppos[i][3]))
                elif i == (num-1):
                    line = "%d,%f,%f,%f" % (int(ppos[i][0]), float(ppos[i][1]), float(ppos[i][2]), float(ppos[i][3]))
                output.write(line)
        print "[+] Wrote %d particles to \"%s\"" %(i+1, fname)
