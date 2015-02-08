import configuration
from cli import *
import sys


def main():
    cli = Interface()
    particles = cli.setSimRules()
    print particles[0]

if __name__ == '__main__':
    try:
        main()
    except Exception as e:  # catch all exceptions at the last possible chokepoint
        print "[-] %s" % str(e)
