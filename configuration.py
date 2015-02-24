from os import path


class Config():
    
    def __init__(self, config="sph.conf"):
        # List of all keys to check for in the config file
        self.keys = ["num", "maxiter", "bound", "stdev", "t_norm", "x_norm", "kernel", "savefile", "gtype", "smooth"]
        # Dictionary containing key (e.g. infile) and value (e.g. "input.csv")
        self.args = {}
        if path.isfile(config):
            self.parseConfig(config)
        else:
            name = raw_input("[?] No config file found. What would you like to name yours? ")
            self.createConfig(name)

    ######################################################
    # Grabs a config argument value. Key == 'All' returns a dictionary containing all current key-value pairs
    # INPUT: key (which argument to grab from config file)
    # OUTPUT: Either a value matching a key pair, or False, indicating the key does not exist
    ######################################################
    def getArg(self, key):
        if key.lower() == 'all':
            return self.args
        elif key in self.keys:
            return self.args[key]
        else:
            print "[-] Couldn't find %s" % key
            return False

    ######################################################
    # Creates a configuration file with keys from self.keys, and no values defined
    # Should only be called when it's determined that no configuration file is found
    # INPUT: fname (configuration filename to create)
    # OUTPUT: none
    ######################################################
    def createConfig(self, fname):
        if fname is '':
            fname = 'sph.conf'
        print "[+] Creating new config file"
        # Instantiate all keys to None
        with open(name, "w") as conf:
            for key in self.keys:
                line = "%s=%s" % (key, '\n')
                conf.write(line)
                self.args[key] = None

    ######################################################
    # Grabs data from a config file line-by-line as key-value pairs stored in self.args
    # Ignores comments - lines starting with #
    # Also ignores empty lines - lines starting with '\n'
    # INPUT: fname (configuration filename to parse. Default is "sph.conf")
    # OUTPUT: none
    ######################################################
    def parseConfig(self, config="sph.conf"):
        with open(config, "r") as conf:
            lines = conf.readlines()
            for line in lines:
                # skip commented out and empty lines
                if line.startswith("#") or line.startswith("\n"):
                    pass
                else:  # read in everything as 'key=value' with NO newlines or spaces
                    line = line.strip().split(" ")
		    # make sure there is a key-value pair of form 'key=value' to read
		    if "=" not in line[0]:
			raise Exception("Config file is incorrectly formatted. Exiting...")
			sys.exit(0)
                    p = line[0].split("=")
                    # check if the key is in self.keys first to avoid arbitrary key-vals
                    if p[0] in self.keys:
                        self.args[p[0]] = p[1]
