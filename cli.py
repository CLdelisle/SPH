__author__ = 'Thomas'
import argparse
import sys

class Interface():
    def __init__(self):
        parser = argparse.ArgumentParser(description="Run the SPH simulation")
        parser.add_argument("mode", help="Tells the program which mode to run in", choices=["run"])
        parser.add_argument("--numprt", help="Number of particles to generate. Conflicts with [IFILE] argument",
                            type=int)
        parser.add_argument("-i", "--ifile", help="Input file path to read particle from. "
                                                  "Takes precedence over [NUMPRT] argument")
        parser.add_argument("-o", "--ofile", help="Output file path to write particles to")
        parser.add_argument("--maxiter", help="Maximum iterations to run the simulation through. Default is 10"
                            , type=int)
        parser.add_argument("--bound", help="Sets boundaries of particle distance. Default is 100", type=int,
                            default=100)
        parser.add_argument("--t_norm", help="Time normalization", choices=["months, years, decades, centuries"])
        parser.add_argument("--x_norm", help="Space normalization", choices=["Meters, kilometers, light-years"])
        parser.add_argument("--gsc")
        parser.add_argument("--kernel", help="Kernel function to use. Default is Gaussian", default="gaussian")
        parser.add_argument("--vidlen", help="Maximum length (in seconds) of video to produce", type=int)
        parser.add_argument("--resolution", help="Resolution of the simulation to use")
        self.args = parser.parse_args()

    def changeConfig(self):
        print 'changeConfig'

    def setSimRules(self):
        print self.args
        if not self.args.ifile and self.args.numprt:
            print "[+] Generating particles..."
        elif self.args.ifile:
            print "[+] Reading from input file %s" % self.args.ifile
        else:
            print "[-] You did not specify an input file or tell me to generate particles. Exiting..."
            sys.exit(1)