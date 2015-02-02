__author__ = 'Thomas Gardner'
from os import path


class Config():
    def __init__(self, file="sph.conf"):
        # List of all keys to check for in the config file
        self.keys = ["numptr", "ifile", "maxiter", "bound", "t_norm", "x_norm", "gsc", "kernel",
                     "vidlen", "resolution", "ofile", "maxvid"]
        # Dictionary containing key (e.g. infile) and value (e.g. "input.csv")
        # Default values are defined later
        self.args = {}
        if path.isfile(file):
#            print "[+] Found config file: %s" % file
            self.parseConfig(file)
        else:
            name = raw_input("[?] No config file found. What would you like to name yours? ")
            self.createConfig(name)

    def getArg(self, key):
        if key == 'all':
            return self.args
        elif key in self.keys:
            return self.args[key]
        else:
            print "[-] Couldn't find %s" % key

    def createConfig(self, name):
        print "[+] Creating new config file"
        # Instantiate all keys to None
        with open(name, "w") as conf:
            for key in self.keys:
                line = "%s=%s" % (key, '\n')
                conf.write(line)
                self.args[key] = None

    def parseConfig(self, file="sph.conf"):
        with open(file, "r") as conf:
            lines = conf.readlines()
            for line in lines:
                line = line.strip()
                p = line.split("=")
                if p[0] in self.keys:
                    self.args[p[0]] = p[1]
        conf.close()