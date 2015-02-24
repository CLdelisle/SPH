import numpy as np

class Particle(object):
	def __init__(self, id,m,x,y,z,vx,vy,vz):
		self.id = id				# particle id (int)
		self.mass = m 				# particle mass (double)
		self.pos = np.array([x,y,z]) 		# position vector<double>
		self.vel = np.array([vx,vy,vz]) 	# velocity vector<double>
		self.acc = np.array([0.0,0.0,0.0])	# acceleration vector<double>

