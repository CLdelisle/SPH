from os import path
from glob import glob   # GLOB can returns a list of files in a directory matching certain regex

class Config():

    def __init__(self, config="sph.conf"):
        # Dictionary containing key (e.g. infile) and default value (e.g. "save.csv")
        self.args = {"num":"", "maxiter":"100", "bound":"100", "stdev":"10.0", "t_norm":"centuries", "timestep":"10", "mass":"50.0", "verbosity":"2",
        "x_norm":"ly", "kernel":"gaussian", "savefile":"example.csv", "gtype":"random", "smooth":"50.0", "interval":"10", "mode":"serial"}
        # Check for a default config file
        if path.isfile(config):
            print "[+] Using default config file: %s" % config
            self.parseConfig(config)
        else:
            files = glob("*.conf")
            if len(files) == 0:
                name = raw_input("[?] No config file found. What would you like to name yours? ")
                if not name.endswith(".conf"):
                    name = name+".conf"
                self.createConfig(name)
            elif len(files) > 1:
                config = files[0]
                print "[+] Using first config file found in this directory: \"%s\"" % config
                self.parseConfig(config)
            elif len(files) == 1:
                config = files[0]
                print "[+] Using \"%s\" as config file" % config
                self.parseConfig(config)
            else:
# OLD                raise Exception("An error occurred when looking for a .conf file. Does one exist?")
                raise IOError("An error occurred when looking for a configuration file. Does one exist?")

    ######################################################
    # Grabs a config argument value. Key == 'All' returns a dictionary containing all current key-value pairs
    # INPUT: key (which argument to grab from config file)
    # OUTPUT: Either a value matching a key pair, or False, indicating the key does not exist
    ######################################################
    def getArg(self, key):
        try:
                if key.lower() == 'all':
                    return self.args
                elif key in self.args.keys():
                    return self.args[key]
        except:
            line = "Couldn't find %s" % key
            raise IndexError(line)
# OLD            print "[-] Couldn't find %s" % key
# OLD            return False

    ######################################################
    # Creates a configuration file with keys from self.args - a dictionary of keys and values
    # Should only be called when it's determined that no configuration file is found
    # INPUT: fname (configuration filename to create)
    # OUTPUT: none
    ######################################################
    def createConfig(self, fname):
        try:
            if fname == '.conf':
                "[!] No config file name specified. Using sph.conf..."
                fname = 'sph.conf'
            print "[+] Creating new config file: %s" % fname
            # Generate new configuration from default values in self.args
            with open(fname, "w") as conf:
                for key in self.args.keys():
                    if key != 'num':
                        line = "%s=%s\n" % (str(key), self.args[key])
                        conf.write(line)
        except:
            raise IOError("An error occurred when creating the configuration file.")

    ######################################################
    # Grabs data from a config file line-by-line as key-value pairs stored in self.args
    # Ignores comments - lines starting with #
    # Also ignores empty lines - lines starting with '\n'
    # INPUT: fname (configuration filename to parse. Default is "sph.conf")
    # OUTPUT: none
    ######################################################
    def parseConfig(self, config="sph.conf"):
        try:
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
     			        raise Exception
                p = line[0].split("=")
                # check if the key is in self.keys first to avoid arbitrary key-vals
                if p[0] in self.args.keys():
                    self.args[p[0]] = p[1]
        except:
            raise Exception("Config file is incorrectly formatted.")
