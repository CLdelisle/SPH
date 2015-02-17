#!/usr/bin/env python

"""
This is the framework for iterating over a list of particles, computing particle accelerations, and numerically integrating their equations of motion.
"""

__author__ = "Colby"
__version__ = "1.0.1"

""" THIS VERSION HAS NOT BEEN TESTED, BUT RUNS SUCCESSFULLY AT THE VERY LEAST """


import numpy as np


class Particle(object):
	def __init__(self,id,m,x,y,z,vx,vy,vz):
		self.id = id	# particle id (int)
		self.mass = m		# particle mass (double)
		self.pos = np.array([x,y,z])			# position vector<double>
		self.vel = np.array([vx,vy,vz])		# velocity vector<double>
		self.acc = np.array([0.0,0.0,0.0])	# acceleration vector<double>


def Newtonian_gravity(p,q):
	# Newton's gravitational constant
	CONST_G = 6.67384 # * 10^(-11) m^3 kg^-1 s^-2
	
	'''
	F = (m_p)a = G(m_p)(m_q)(r)/r^3 -> a = (G * m_q)(r)/(g(r,r)^(3/2)), with g(_,_) the Euclidian inner product
	Note that this is all in the r-direction vectorially
	'''

	r = q.pos - p.pos # separation vector
	R = np.sqrt(r.dot(r)) # magnitude of the separation vector
	return ((CONST_G * q.mass) / (R**3)) * r


def main():
	CONST_H = 1.0 # size of timestep (this should come from config file)
	CONST_T_MAX = 10 # max iteration time (this should come from config file)
	t = 0.0 # elapsed time
	particles = []
	particles.append(Particle(1,1.0,2.0,2.0,2.0,3.0,3.0,3.0))
	particles.append(Particle(2,1.0,-2.0,-2.0,-2.0,3.0,3.0,3.0))
	particles.append(Particle(3,1.0,5.0,0.0,-5.0,0.0,0.0,0.0))
	
	'''
	So we can do this one of two ways.
	1) Keep only one copy of the system in memory.
	   This is what is implemented here. This does not require
	   data duplication, but necessitates two iterations through
	   the list.
	2) Keep two lists. One for t = t_N, one for t = t_(N+1).
	   This way we can remove a for-loop, but requires more memory.
	Luckily, it is likely these problems only affect the serial algorithm.
	'''
	
	while(t < CONST_T_MAX):
		# main simulation loop
		for p in particles:
			# preemptively start the Velocity Verlet computation (first half of velocity update part)
			p.vel += (CONST_H/2.0) * p.acc
			temp = p.acc
			p.acc = 0
			for q in particles:
				if(p.id != q.id):
					p.acc += Newtonian_gravity(p,q)
			p.vel += (CONST_H/2.0) * p.acc # finish velocity update
	
		'''
		Velocity Verlet integration: Works only assuming force is velocity-independent
		http://en.wikipedia.org/wiki/Verlet_integration#Velocity_Verlet
		'''
		
		for p in particles:
			# perform position update
			p.pos += CONST_H * (p.vel + (CONST_H/2.0)*temp)

		t += CONST_H # advance time

		print "Iteration " + str(int(t)) + "..." # Output only to verify that this code is syntactically correct and will run


if __name__ == '__main__':
    try:
        main()
    except Exception as e:  # catch all exceptions at the last possible chokepoint - stole from Thomas' code
        print "[-] %s" % str(e)
