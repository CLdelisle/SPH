from cli import *

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