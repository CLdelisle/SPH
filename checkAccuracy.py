from cli import *

def main():
  cli = Interface()
  cli.args.ifile = 'example.csv'
  ppos = cli.readInputFile()
  getTotalMassAndMomentum(ppos)

def getTotalMassAndMomentum(particles):
  for particle in particles:
  	print particle.vel
main()