__author__ = 'Thomas Gardner'
from os import path


class Config():
    def __init__(self, file="sph.conf"):
        # List of all keys to check for in the config file
        self.keys = ["numptr", "savefile", "maxiter", "bound", "stdev", "t_norm", "x_norm", "gsc", "kernel",
                     "vidlen", "res", "ofile", "maxvid"]
        # Dictionary containing key (e.g. infile) and value (e.g. "input.csv")
        self.args = {}
        if path.isfile(file):
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
            return False

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
                # skip commented out and empty lines
                if line.startswith("#") or line.startswith("\n"):
                    pass
                else:  # read in everything as 'key=value'
                    line = line.strip()
                    p = line.split("=")
                    # check if the key is in self.keys first to avoid arbitrary key-vals
                    if p[0] in self.keys:
                        self.args[p[0]] = p[1]
