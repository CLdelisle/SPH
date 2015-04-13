from particle import Particle
import numpy as np

"""
This is the framework for iterating over a list of particles, computing particle accelerations, and numerically integrating their equations of motion.
"""

__author__ = "Colby"
__version__ = "2.0.0"

""" THIS VERSION HAS NOT BEEN TESTED, BUT RUNS SUCCESSFULLY AT THE VERY LEAST """

# Params p,q particles
def Newtonian_gravity(p,q):
	# Newton's gravitational constant
	CONST_G = 6.67384 # * 10^(-11) m^3 kg^-1 s^-2
	
	'''
	F = (m_p)a = G(m_p)(m_q)(r)/r^3 -> a = (G * m_q)(r)/(g(r,r)^(3/2)), with g(_,_) the Euclidian inner product
	Note that this is all in the r-direction vectorially
	'''

	zz = q.pos - p.pos # separation vector
	R = np.linalg.norm(zz) # magnitude of the separation vector
	return ((CONST_G * q.mass) / (R**3)) * zz


def find_kernel(x, r, h):
	# if 1 (true) use Gaussian
	# if 0 (false) use spline
	if(x):
		return Gaussian_kernel(r, h)
	else:
		return cubic_spline_kernel(r, h)


def del_kernel(x, r, h):
	# if 1 (true) use Gaussian
	# if 0 (false) use spline
	if(x):
		return del_Gaussian(r, h)
	else:
		return del_cubic_spline(r, h)


def Gaussian_kernel(r, h):
	# Gaussian function
	r = np.linalg.norm(r)

	return ( (((1/(np.pi * (h**2)))) ** (3/2) ) * ( np.exp( - ((r**2) / (h**2)) )) )


def del_Gaussian(r, h):
	# derivative of Gaussian kernel
	r1 = np.linalg.norm(r)
	return ( ((-2 * r1) / (h**2)) * Gaussian_kernel(r, h))


def cubic_spline_kernel(r, h):
	# cubic spline function - used if one needs compact support
	return 0.5 # this is a bullshit placeholder

def del_cubic_spline(r, h):
	# derivative of cubic spline
	return 0.5 # this is a bullshit placeholder

def pressure(p):
	k = 1.0 #this may need to stay hardcoded for our purposes, though could be read in from config file
	gamma = 1.5 #but i'm keeping these constants segregated in this function for now instead of inlining because of this issue
	return (k * (p.rho ** gamma))

def saveParticles(particles, fname):
#	if particles:
	        fhandle = open(fname, "w")
	        for p in particles:
	                p.writeToFile(fhandle)
	        fhandle.close()
#	else:
#		print "[-] No more particles in list!"


def sim(particles, bound, kernel, maxiter, pnum, smooth, t_norm, x_norm, interval, savefile, timestep, mode):
	t = 0.0     # elapsed time
	if(kernel == "gaussian"):
		CHOOSE_KERNEL_CONST = 1
	else:
		CHOOSE_KERNEL_CONST = 0

	if mode == "parallel":
		import pycuda.driver as cuda
		import pycuda.autoinit
		import numpy
		from pycuda.compiler import SourceModule

		from gpu_interface import ParticleGPUInterface
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
        # output-100.csv = prefix + interval + file extension
	ary = savefile.split(".")  # only split savefile once ([0]=prefix, [1]=extension)
	save = 0
#	print "[+] Saved @ iterations: ",
	while(t < (maxiter*timestep)):
			if (save*interval) == t:
					fname = "%s-%d.%s" % (ary[0], int(t), ary[1])
					save += 1  # bump save counter
				#	string = "\b%d..." % int(t)     # '\b' prints a backspace character to remove previous space
				#	print string,
					saveParticles(particles, fname)

			# main simulation loop
			if mode == "parallel":
				# init gpu interface, pass particles
				gpu_particles = ParticleGPUInterface(particles)

				gpu_particles.sim_loop("run_simulation_loops", timestep, smooth, CHOOSE_KERNEL_CONST)
				# Transfer the results back to CPU
				# Just for testing, this should not be done here
				particles = gpu_particles.getResultsFromDevice()

			else:
				# first sim loop (could use a better name, but I have no idea what this loop is doing)
				for p in particles:
						# preemptively start the Velocity Verlet computation (first half of velocity update part)
						p.vel += (timestep/2.0) * p.acc
						p.temp = p.acc
						p.acc = 0.0
						p.rho = 0.0
						p.pressure = 0.0
						#get density
						for q in particles:
					#	        print find_kernel(CHOOSE_KERNEL_CONST, p.pos - q.pos, smooth)
							p.rho += ( q.mass * (find_kernel(CHOOSE_KERNEL_CONST, p.pos - q.pos, smooth)) )
							# while we're iterating, add contribution from gravity
							if(p.id != q.id):
								p.acc += Newtonian_gravity(p,q)
						# normalize density
						p.rho = ( p.rho / len(particles) )
						p.pressure = pressure(p)

				# second sim loop
				for p in particles:
					# acceleration from pressure gradient
					for q in particles:
					        if p.id != q.id:
	        					p.acc -= ( q.mass * ((p.pressure / (p.rho ** 2)) + (q.pressure / (q.rho ** 2))) * del_kernel(CHOOSE_KERNEL_CONST, p.pos - q.pos, smooth) ) * (1 / (np.linalg.norm(p.pos - q.pos))) * (p.pos - q.pos)
					# finish velocity update
	                                p.vel += (timestep/2.0) * p.acc
				'''
				Velocity Verlet integration: Works only assuming force is velocity-independent
				http://en.wikipedia.org/wiki/Verlet_integration#Velocity_Verlet
				'''
				# iterate AGAIN to do final position updates
				# save particles list to temporary holder - ensures we have consistent indexing throughout for loop
				#	tempp = particles

				# third sim loop
				for p in particles:
					# perform position update
					p.pos += timestep * (p.vel + (timestep/2.0)*p.temp)
			#                if np.linalg.norm(p.pos) > bound:
			#                        print "Particle %d position: %f out of range at iteration %d" % (p.id, np.linalg.norm(p.pos), int(t))
			#                        tempp.remove(p)
			#        particles = tempp
			t += timestep  # advance time

	# Always save the last interval
#	print "\b%d\n" % int(t)
        print
	fname = "%s-%d.%s" % (ary[0], int(t), ary[1])
	saveParticles(particles, fname)
	return t    # returns the last t-value, which is useful for displaying total iterations
