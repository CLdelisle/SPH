import argparse
from sys import exit
import configuration
from random import uniform, gauss
from particle import Particle as particle
import framework

class Interface():

    def __init__(self):
        self.config = configuration.Config()
        defaults = self.config.getArg('all')
        # Set argument options for CLI
        self.parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description="SPH.py is the user interface for "
                                "our physics simulation program.\n'SPH' stands for Smoothed Particle Hydrodynamics, which is an algorithm for "
                                "simulating fluid flows.\nYou can read more on it at https://en.wikipedia.org/wiki/Smoothed-particle_hydrodynamics")

        self.parser.add_argument("-g", "--gen", help="Number of particles to generate. Conflicts with [IFILE] argument", type=int)
        self.parser.add_argument("-i", "--ifile", help="Input file path to read particles from. Takes precedence over [GEN] argument")
        self.parser.add_argument("--gtype", help="Type of particle generation to perform. Default is "+defaults['gtype'], choices=['gaussian', 'random'], default=defaults['gtype'])
        self.parser.add_argument("-s", "--savefile", help="Output file path to write particles to. Suffix is currently "+defaults['savefile'], default=defaults['savefile'])
        self.parser.add_argument("--bound", help="Sets boundaries of particle space. Default is "+defaults['bound'], type=int, default=int(defaults['bound']))
        self.parser.add_argument("--stdev", help="Standard deviation of particle space. Default is "+defaults['stdev'], type=float, default=float(defaults['stdev']))
        self.parser.add_argument("--maxiter", help="Maximum iterations to run the simulation through. Default is "+defaults['maxiter'], type=int, default=int(defaults['maxiter']))
        self.parser.add_argument("--timestep", help="The temporal resolution of the simulation. Default is "+defaults['timestep'], type=int, default=defaults['timestep'])
        self.parser.add_argument("--t_norm", help="Time normalization. Default is "+defaults['t_norm'], choices=['months', 'years', 'decades', 'centuries'], default=defaults['t_norm'])
        self.parser.add_argument("--x_norm", help="Space normalization. Default is "+defaults['x_norm'], choices=['m', 'km', 'ly'], default=defaults['x_norm'])
        self.parser.add_argument("--kernel", help="Kernel function to use. Default is "+defaults['kernel'], choices=['gaussian', 'cubic'], default=defaults['kernel'])
        self.parser.add_argument("--smooth", help="Smoothing for the kernel function. Default is "+defaults['smooth'], type=float, default=float(defaults['smooth']))
        self.parser.add_argument("--interval", help="How many loops before particles are saved. Default is "+defaults['interval'], type=int, default=int(defaults['interval']))
	self.parser.add_argument("--mass", help="Mass of the particles in the simulation. Default is "+defaults['mass'], type=float, default=float(defaults['mass']))
        # Actually begin to parse the arguments
        self.args = self.parser.parse_args()

    ######################################################
    # Generate a specified number of particles in a 3D space using random.gauss() function
    # Should cluster particles in a sort of 'globe' in the center of the grid
    # INPUT: num (number of particles to generate), mass (default mass of particles)
    # OUTPUT: ppos (array containing particle objects)
    ######################################################
    def createGaussian(self, num, mass):
        mean = (self.args.bound/2)
        ppos = [] # array containing particle positions
        
        for i in range(0, num):
            # coordinate values for X, Y, and Z
            coords = [-1, -1, -1]  # start each coordinate at -1 to ensure the loop runs once (Damn you Python for not having a Do-While)

            for j in range(0, 3):
                while coords[j] < 0 or self.args.bound < coords[j]: # ensure val is within range of particle boundaries
                    coords[j] = round(gauss(mean, self.args.stdev), 6)

            m = round(float(mass), 2)
	# Set initial velocities to 0.0 for TESTING ONLY
            vx = 0.0
            vy = 0.0
            vz = 0.0
      #      vx = uniform(1,100)
      #      vy = uniform(1,100)
      #      vz = uniform(1,100)
            # Add new particle to ppos with no initial velocity
            # particle(id, m, x, y, z, vx, vy, vz)
            ppos.append(particle(i, m, coords[0], coords[1], coords[2], vx, vy, vz))

        return ppos

    ######################################################
    # Generate a specified number of particles completely randomly
    # Given enough particles, the space should be about equally distributed
    # INPUT: num (number of particles to generate), mass (default mass of particles)
    # OUTPUT: ppos (array containing particle objects)
    ######################################################
    def createRandom(self, num, mass):
        bound = self.args.bound
        ppos = [] # array containing particle positions

        for i in range(0, num):
            # Particle created of the form: [PID, X, Y, Z, M]
            x = round(uniform(0, bound), 6)
            y = round(uniform(0, bound), 6)
            z = round(uniform(0, bound), 6)
            m = round(float(mass), 2)
	# Set initial velocities to 0.0 for TESTING ONLY
            vx = 0.0
            vy = 0.0
            vz = 0.0
        #    vx = uniform(1,100)
        #    vy = uniform(1,100)
        #    vz = uniform(1,100)
            # Add new particle to ppos with no initial velocity
            # particle(id, m, x, y, z, vx, vy, vz)
            ppos.append(particle(i, m, x, y, z, vx, vy, vz))

        return ppos

    ######################################################
    # Will determine what kind of particle generation will occur
    # Mainly just a wrapper function for calling createRandom() or createGaussian()
    # INPUT: num (number of particles to generate), method (how to generate particles)
    # OUTPUT: ppos (array containing particle objects)
    ######################################################
    def genParticles(self, num, method):
        mass = self.args.mass

        if method == 'gaussian':
            print "[+] Generating particles with %s%s distribution in a %s%s^3 space\n" % (str(self.args.stdev), self.args.x_norm, str(self.args.bound), self.args.x_norm)
            ppos = self.createGaussian(num, mass)
        elif method == 'random':
            print "[+] Spreading particles randomly within %s%s^3 space" % (str(self.args.bound), self.args.x_norm)
            ppos = self.createRandom(num, mass)
        else:
            print "[+] No particle generation method selected. Spreading particles randomly"
            ppos = self.createRandom(num, mass)
        
        return ppos

    ######################################################
    # Called when --ifile flag is set
    # Reads from an input file (orly?) of form "PID,X-coord,Y-coord,Z-coord" on each line
    # INPUT: none - grabs input filename from args parameter
    # OUTPUT: ppos (array containing particle objects)
    ######################################################
    def readInputFile(self):
        file = self.args.ifile
        # particle positions - just like self.genParticles
        ppos = []
        # velocity vectors
        try:
            with open(file, "r") as ifile:
               print "[+] Reading from input file \"%s\"" % file
               pstrings = ifile.readlines()
               for i in range(0, len(pstrings)):
                  # header = "ID, mass, px,py,pz\n"
                  # strip '\n' from line, then split into a list at commas
                  p = pstrings[i].strip().split(",")
                  # must cast values, because they are read in as strings by Python
                  # casting should catch odd values as well (e.g. '40a' for a PID)
                  ppos.append(particle(int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4]), float(p[5]), float(p[6]), float(p[7])))
        except:
            raise Exception("[-] Cannot find input file! Does it exist?")

        return ppos

    ######################################################
    # This is where all the CLI rules will be set and interpreted, and further function calls done
    # Contradicting options are corrected here (e.g. --ifile and --gen flags set)
    # Does NOT do any direct data processing. Only calls other functions
    # INPUT: none
    # OUTPUT: particles (array of particles with form [PID, mass, X-coord, Y-coord, Z-coord, Vx, Vy, Vz])
    ######################################################
    def setSimRules(self):
        if self.args.ifile: # If [IFILE] is specified, ignore everything else
            particles = self.readInputFile()
            
        elif not self.args.ifile and self.args.gen: # If [IFILE] isn't specified and [NUMPRT] is specified, generate particles
            particles = self.genParticles(self.args.gen, self.args.gtype)

        else: # If [IFILE] and [NUMPRT] are NOT specified, print help message and exit
            self.parser.print_help()
            print "\n[-] You did not specify an input file or tell me to generate particles!"
            exit(1)

        # retrieved from either an input file or generated in self.genParticles
        return particles

    ######################################################
    # Passes necessary arguments to the simulation
    # INPUT: particles (array of particles with form [PID, mass, X-coord, Y-coord, Z-coord, Vx, Vy, Vz])
    # OUTPUT: none
    ######################################################
    def startSimulation(self, particles):
        print "\n[+] Starting simulation..."
        iterations = framework.sim(particles, self.args.bound, self.args.kernel, self.args.maxiter,
                                self.args.gen, self.args.smooth, self.args.t_norm, self.args.x_norm,
                                self.args.interval, self.args.savefile, self.args.timestep)
