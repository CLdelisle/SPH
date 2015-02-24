#!/usr/bin/python
from cli import *


def main():
    cli = Interface()
    particles = cli.setSimRules()
    print particles[ len(particles)-1 ]  # print out last particle. Hopefully the PID is the right number

if __name__ == '__main__':
    try:
        main()
    except Exception as e:  # catch all exceptions at the last possible chokepoint
        print "[-] %s" % str(e)
