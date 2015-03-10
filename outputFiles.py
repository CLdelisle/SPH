import os

# This file handles the common operations of finding output files, sorting them
# by timestep, and extracting the timesteps as integers

# Takes a prefix, gets all files with that prefix, sorts them, and returns their file names as an array
def sortedFileNames(prefix):
  allOutputFiles = [filename for filename in os.listdir('.') if filename.startswith(prefix) and filename.endswith(".csv")]
  timesteps = [getTimestep(f, prefix) for f in allOutputFiles] # timesteps is an array of the extracted times from filenames [100, 200, 1]
  timesteps.sort()
  return [prefix + "-" + str(n) + '.csv' for n in timesteps]

# Takes a filename and a prefix, and returns the timestep as an int
def getTimestep(filename, prefix):
  return int(filename.replace(prefix + '-', '').replace('.csv', ''))