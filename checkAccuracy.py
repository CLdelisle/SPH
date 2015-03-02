from cli import *
import os

def main():
  cli = Interface()
  cli.args.ifile = 'example.csv'
  ppos = cli.readInputFile()
  print getTotalMassAndMomentum(ppos)

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


# Takes an array of filenumbers, sorts them, and adds in the prefix
def filenamesInOrder(filenumbers, prefix):
  filenumbers.sort()
  return [prefix + "-" + str(n) + '.csv' for n in filenumbers]