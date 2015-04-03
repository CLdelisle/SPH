import numpy as np
from math import sqrt

class Particle(object):
	def __init__(self, id, m, x, y, z, vx, vy, vz):
		self.id = id						# particle id (int)
		self.mass = m						# particle mass (double)
		self.pos = np.array([x,y,z])		# position vector<double>
		self.vel = np.array([vx,vy,vz])		# velocity vector<double>
		self.acc = np.array([0.0,0.0,0.0])	# acceleration vector<double>
		self.rho = 0.0
		self.pressure = 0.0

    # Display the attributes of a given particle - ID, Mass, Position Vector, Velocity Vector, Acceleration Vector
	def display(self, tabs=0):
		p = self.pos
		v = self.vel
		a = self.acc
		if tabs:
			t = "\t"*int(tabs)
		else:
			t = ''
		#	id  mass  posx, posy, posz      vx, vy, vz       ax, ay, az
		print "%s%s\t%.2f\t%.2f, %.2f, %.2f\t%.2f, %.2f, %.2f\t%.2f, %.2f, %.2f\t%.8f\t%.8f" % (t, str(self.id), self.mass, p[0],p[1],p[2], v[0],v[1],v[2], a[0],a[1],a[2], self.pressure, self.rho)

	def velocityMagnitude(self):
		return sqrt(self.vel[0] ** 2 + self.vel[1] ** 2 + self.vel[2] ** 2)

	#  Requires current an open file handle
	def writeToFile(self, output):
		line = self.formatProperties()
		output.write(line)

	def formatProperties(self):
		#   "Particle ID, X-coord, Y-coord, Z-coord, etc."
		return "%d,%.2f,%f,%f,%f,%f,%f,%f,%f,%f\n" % (int(self.id), float(self.mass), float(self.pos[0]), float(self.pos[1]),
				float(self.pos[2]),float(self.vel[0]),
				float(self.vel[1]), float(self.vel[2]),
				float(self.pressure), float(self.rho))

	def flatten(self):
		return [self.id, self.mass, self.pos[0], self.pos[1], self.pos[2]]