from cli import *
import os

def main(prefix = 'output'):
  allOutputFiles = getOutputFiles(prefix)
  timesteps      = getOutputFileNumbers(allOutputFiles, prefix)
  filenames      = fileNamesInOrder(timesteps, prefix)
  return filenames



# Given a filename (output-001.csv), return the total mass and momentum
def totalsFromFile(filename):
  cli = Interface()
  cli.args.ifile = filename
  ppos = cli.readInputFile()
  return getTotalMassAndMomentum(ppos)


# Given an array of particles, return the total mass and momentum
def getTotalMassAndMomentum(particles):
  momentum = mass = 0
  for particle in particles:
    mass += particle.mass
    momentum += mass * particle.velocityMagnitude()
  return (momentum, mass)

# Given a prefix like "output", it finds all files in the current directory with the prefix
def getOutputFiles(prefix):
  return [filename for filename in os.listdir('.') if filename.startswith(prefix)]

# Takes an array of filenames ['output-100.csv', 'output-001.csv']
# Returns the numbers [100, 1]
def getOutputFileNumbers(filenames, prefix):
  return [int(f.replace(prefix+'-', '').replace('.csv', '')) for f in filenames]

# Takes an array of file numbers, sorts them, and adds in the prefix
def fileNamesInOrder(file_numbers, prefix):
  file_numbers.sort()
  return [prefix + "-" + str(n) + '.csv' for n in file_numbers]


print main()