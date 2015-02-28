import numpy as np

class Particle(object):
	def __init__(self, id, m, x, y, z, vx, vy, vz):
		self.id = id				# particle id (int)
		self.mass = m 				# particle mass (double)
		self.pos = np.array([x,y,z]) 		# position vector<double>
		self.vel = np.array([vx,vy,vz]) 	# velocity vector<double>
		self.acc = np.array([0.0,0.0,0.0])	# acceleration vector<double>

	def display(self, tabs=0):
		p = self.pos
		v = self.vel
		a = self.acc
		if tabs:
			t = "\t"*int(tabs)
		else:
			t = ''
		# id    mass    posx, posy, posz    vx, vy, vz      ax, ay, az
		print "%s%d\t%.2f\t%.2f, %.2f, %.2f\t%.2f, %.2f, %.2f\t%.2f, %.2f, %.2f" % (t, self.id, self.mass, p[0],p[1],p[2], v[0],v[1],v[2], a[0],a[1],a[2])
