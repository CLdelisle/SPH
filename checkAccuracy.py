from cli import *
from outputFiles import *

def main(prefix = 'output'):
  sorted_filenames = sortedFileNames(prefix)
  for x in calculatePercentChanges(sorted_filenames, prefix):
          print x

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

if __name__ == '__main__':
  main()
