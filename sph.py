#!/usr/bin/python
from cli import *


def main():
    cli = Interface()
    ppos = cli.setSimRules()
    print "\tid\tmass\tposx, posy, posz\tvx, vy, vz\t\tax, ay, az"
    for particle in ppos:
        particle.display(tabs=1)
    
    cli.startSimulation()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:  # catch all exceptions at the last possible chokepoint
        print "[-] %s" % str(e)
