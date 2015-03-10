from mpl_toolkits.mplot3d import axes3d
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from outputFiles import sortedFileNames
import argparse

parser = argparse.ArgumentParser(description='Create a 3D particle plot')
parser.add_argument("--prefix", help="Output file prefix. Defaults to 'output'.",
	                        default='output')
args = parser.parse_args()

files = sortedFileNames(args.prefix)

if (len(files) == 0):
	raise ValueError("No output csv files found with the prefix '{}'".format(args.prefix))

RUNTIME = len(files) - 1 #Number of frames to render

curr_x = []
curr_y = []
curr_z = []

data = []

# data file format: Particle ID, X-coord, Y-coord, Z-coord, X-Velocity, Y-Velocity, Z-Velocity
for i, filename in enumerate(files):
	data.append(np.genfromtxt(filename, delimiter=',', names=['pid', 'mass', 'x', 'y', 'z', 'vx', 'vy', 'vz']))

t = 0
def update(num,sc,ax):
	ax.cla();
	global t
	global data
	global curr_x
	global curr_y
	global curr_z

	curr_x = []
	curr_y = []
	curr_z = []
	
	for particle in range(len(data[0]['x'])):
		curr_x.append(data[t]['x'][particle])
		curr_y.append(data[t]['y'][particle])
		curr_z.append(data[t]['z'][particle])

	t += 1
	ax.autoscale(False)
	sc = ax.scatter(curr_x, curr_y, curr_z, c='m', marker='o')
	return sc

def main():
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	curr_x = []
	curr_y = []
	curr_z = []
	for particle in range(len(data[0]['x'])):
		curr_x.append(data[0]['x'][particle])
		curr_y.append(data[0]['y'][particle])
		curr_z.append(data[0]['z'][particle])
		
		
	ax.set_xlim3d(-700,700);
	ax.set_ylim3d(-700,700);
	ax.set_zlim3d(-700,700);

	sc = ax.scatter(curr_x, curr_y, curr_z, c='r', marker='o')

	ani = animation.FuncAnimation(fig, update, frames=RUNTIME,fargs=(sc,ax))
	
	ani.save('test_run.gif', writer='imagemagick', fps=15);
	# ani.save('test_run.mp4', fps=15, extra_args=['-vcodec', 'libx264'])
	# ani.save("demo.avi",codec='avi')
	# ani.save('test_run.mp4', fps=3,extra_args=['-vcodec', 'h264','-pix-fmt', 'yuv420p'])
	
	# plt.show()
	# plt.close()
	return

if __name__ == '__main__':
	main()