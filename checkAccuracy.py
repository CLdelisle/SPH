from cli import *
import os

def main(prefix = 'output'):
  allOutputFiles        = getOutputFiles(prefix)
  timesteps             = getOutputFileNumbers(allOutputFiles, prefix)
  sorted_filenames      = fileNamesInOrder(timesteps, prefix)

  print calculatePercentChanges(sorted_filenames, prefix)

# Given a filename (output-1.csv), return the total mass and momentum
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

# Takes an array of filenames ['output-100.csv', 'output-1.csv']
# Returns the numbers [100, 1]
def getOutputFileNumbers(filenames, prefix):
  return [getTimestep(f, prefix) for f in filenames]

def getTimestep(filename, prefix):
	return int(filename.replace(prefix + '-', '').replace('.csv', ''))

# Takes an array of file numbers, sorts them, and adds in the prefix
def fileNamesInOrder(file_numbers, prefix):
  file_numbers.sort()
  return [prefix + "-" + str(n) + '.csv' for n in file_numbers]

# Returns array of tuples [(percent_change_momentum, percent_change_mass, timestep), ...]
def calculatePercentChanges(filenames, prefix):
  # the other file's accuracy are based on the first timestep (original particle locations)
  perfect_accuracy_data = totalsFromFile(filenames[0])
  return map(lambda f: filePercentChanges(f, perfect_accuracy_data) + (getTimestep(f, prefix),), filenames[1:])

# helper method for calculatePercentChanges
def filePercentChanges(filename, max_accuracy):
  perfect_mass = max_accuracy[1]
  perfect_momentum = max_accuracy[0]

  file_momentum, file_mass = totalsFromFile(filename)
  return (percentChange(file_momentum, perfect_momentum), percentChange(file_mass, perfect_mass))

# Helper method that calculates percent change
def percentChange(newValue, oldValue):
  return (float(newValue) - float(oldValue)) / float(oldValue) * 100

main()