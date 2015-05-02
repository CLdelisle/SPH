import numpy as np
from math import sqrt

class Particle(object):
	def __init__(self, id, m, x, y, z, vx, vy, vz, acc=None, rho=None, pre=None, temp=None):
		self.id = id				# particle id (int)
		self.mass = m				# particle mass (double)
		self.pos = np.array([x,y,z])		# position vector<double>
		self.vel = np.array([vx,vy,vz])		# velocity vector<double>
		
		if temp is None:
			self.temp = np.array([0.0,0.0,0.0])	# temp vector<double>
		else:
			self.temp = np.array(temp) # force temp to be a numpy array
		
		if acc is None:
			self.acc = np.array([0.0,0.0,0.0])	# acceleration vector<double>
		else:
			self.acc = np.array(acc) # force acc to be a numpy array
		if rho is None:
			self.rho = 0.0
		else:
			self.rho = rho
		if pre is None:
			self.pressure = 0.0
		else:
			self.pressure = pre

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
	def writeToFile(self, output, v):
		line = self.formatProperties(v)
		output.write(line)

	#   "Particle ID, X-coord, Y-coord, Z-coord, etc."
	def formatProperties(self, v):
		if v == 1: return "%.3f,%.3f,%.3f\n" % (float(self.pos[0]), float(self.pos[1]), float(self.pos[2]))
		if v == 2: return "%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f, %.2f, %d\n" % (float(self.pos[0]), float(self.pos[1]), float(self.pos[2]),float(self.vel[0]), float(self.vel[1]), float(self.vel[2]), float(self.pressure), float(self.rho), float(self.mass), int(self.id))
		if v == 3: return "%f,%f,%f,%f,%f,%f,%f,%f, %.2f, %d\n" % (float(self.pos[0]), float(self.pos[1]), float(self.pos[2]),float(self.vel[0]), float(self.vel[1]), float(self.vel[2]), float(self.pressure), float(self.rho), float(self.mass), int(self.id))

	def flatten(self):
		return [self.id, self.mass, self.pos[0], self.pos[1], self.pos[2], self.vel[0], self.vel[1], self.vel[2], self.acc[0], self.acc[1], self.acc[2], self.rho, self.pressure, self.temp[0], self.temp[1], self.temp[2]]

	@staticmethod
	def unflatten(float_array):
		return Particle(float_array[0], float_array[1], float_array[2], float_array[3], float_array[4], float_array[5], float_array[6], float_array[7], [float_array[8], float_array[9], float_array[10]], float_array[11], float_array[12], [float_array[13], float_array[14], float_array[15]])

