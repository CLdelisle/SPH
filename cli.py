__author__ = 'Thomas'
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
        parser = argparse.ArgumentParser(description="Run the SPH simulation")
        parser.add_argument("-g", "--gen", help="Number of particles to generate. Conflicts with [IFILE] argument",
                            type=int)
        parser.add_argument("-i", "--ifile", help="Input file path to read particles from. "
                                "Takes precedence over [GEN] argument")
        parser.add_argument("-o", "--ofile", help="Output file path to write particles to. Suffix is currently "+defaults['ofile'])
        parser.add_argument("--bound", help="Sets boundaries of particle space. Default is "+defaults['bound'],
                            type=int, default=int(defaults['bound']))
        parser.add_argument("--stdev", help="Standard deviation of particle space. Default is "+defaults['stdev'],
                    type=float, default=float(defaults['stdev']))
        parser.add_argument("--maxiter", help="Maximum iterations to run the simulation through. Default is "+defaults['maxiter'],
                           type=int, default=int(defaults['maxiter']))
        parser.add_argument("--t_norm", help="Time normalization. Default is "+defaults['t_norm'],
                            choices=["months, years, decades, centuries"], default=defaults['t_norm'])
        parser.add_argument("--x_norm", help="Space normalization. Default is "+defaults['x_norm'],
                            choices=["Meters, kilometers, light-years"], default=defaults['x_norm'])
        parser.add_argument("--gsc", default=defaults['gsc'])
        parser.add_argument("--kernel", help="Kernel function to use. Default is "+defaults['kernel'],
                            default=defaults['kernel'])
        parser.add_argument("--vidlen", help="Maximum length (in seconds) of video to produce", type=int)
        parser.add_argument("--res", help="Resolution of the simulation to use", default=defaults['res'])
        self.args = parser.parse_args()

    def genParticles(self, num, kernel):
        if kernel == 'gaussian':
            print "[+] Using Gaussian kernel distribution"
            ppos = self.useGaussian(num)
        else:
            "[+] What the fuck kind of kernel are you using?"
            sys.exit(0)
        
        # takes filename specified in sph.config, particle positions, and number of particles generated
        self.writeParticlesToFile(self.config.getArg('savefile'), ppos, num)
        return ppos
    
    def readInputFile(self):
        # particle positions - just like self.genParticles
        ppos = []
        print "[+] Reading from input file \"%s\"" % self.args.ifile
        with open(self.args.ifile, "r") as ifile:
            for line in ifile:
                # header = "Particle ID, X-coord, Y-coord, Z-coord\n"
                # strip '\n' from line, then split into a list at commas
                ppos.append(line.strip().split(","))
        return ppos

    def setSimRules(self):
        
        if self.args.ifile: # If [IFILE] is specified, ignore everything else
            particles = self.readInputFile()
        elif not self.args.ifile and self.args.gen: # If [IFILE] isn't specified and [NUMPRT] is specified, generate particles
            particles = self.genParticles(self.args.gen, self.args.kernel)
        else: # If [IFILE] and [NUMPRT] are NOT specified, exit
            print "[-] You did not specify an input file or tell me to generate particles. Exiting...\n"
            sys.exit(1)

        # retrieved from either an input file or generated in self.genParticles
        return particles

    def useGaussian(self, num):
        mean = (self.args.bound/2)
        ppos = [] # array containing particle positions
        for i in range(0, num):
            particle = []
            particle.append(i) # assign particle id
            val = -1 # coordinate value for X, Y, or Z
            for j in range(1, 4):
                while val < 0 or self.args.bound < val: # ensure val is within range of particle boundaries
                    val = gauss(mean, self.args.stdev)
                particle.append(val)
                val = -1  # reset val so the while loop will run again (otherwise X=Y=Z)
            ppos.append(particle)
            
        return ppos

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
