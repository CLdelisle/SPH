from mpl_toolkits.mplot3d import axes3d
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from outputFiles import sortedFileNames
import argparse

parser = argparse.ArgumentParser(description='Create a 3D particle plot')
parser.add_argument("--prefix", help="Output file prefix. Defaults to 'output'.", default='output')
parser.add_argument("--fps", help="Frames per second Defaults to 15", type=int, default=15)
parser.add_argument("--file", help="Filename to save animation to", default='plot.gif')
parser.add_argument("--title", help="Plot title (defaults to blank)", default='')
parser.add_argument("--bound", help="Bound of the simulation used", default=1000, type=int)
parser.add_argument("--rotation", help="Speed of the rotation", default=1.0, type=float)
args = parser.parse_args()

files = sortedFileNames(args.prefix)

if (len(files) == 0):
	raise ValueError("No output csv files found with the prefix '{}'".format(args.prefix))


data = []

# data file format: Particle ID, X-coord, Y-coord, Z-coord, X-Velocity, Y-Velocity, Z-Velocity
for i, filename in enumerate(files):
	data.append(np.genfromtxt(filename, delimiter=',', names=['x', 'y', 'z']))

RUNTIME = len(files) - 1 #Number of frames to render

# Returns arrays (x,y,z) for all particles at a given time, defaults to initial configuration
def get_particle_positions(time=0):
	curr_x, curr_y, curr_z = [], [], []

	for particle in range(len(data[time]['x'])):
		curr_x.append(data[time]['x'][particle])
		curr_y.append(data[time]['y'][particle])
		curr_z.append(data[time]['z'][particle])

	return curr_x, curr_y, curr_z

t = 0
def update(num, sc, ax, rotationSpeed):
	ax.cla();
	global t

	curr_x, curr_y, curr_z = get_particle_positions(t)

	t += 1
	ax.autoscale(False)
	ax.view_init(elev=10, azim=t*rotationSpeed)
	sc = ax.scatter(curr_x, curr_y, curr_z, c='m', marker='o')
	return sc

def main():
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	if args.title != "":
		fig.suptitle(args.title, fontsize=20)

	curr_x, curr_y, curr_z = get_particle_positions()

	ax.set_xlim3d((-1*args.bound), args.bound)
	ax.set_ylim3d((-1*args.bound), args.bound)
	ax.set_zlim3d((-1*args.bound), args.bound)

	sc = ax.scatter(curr_x, curr_y, curr_z, c='r', marker='o')

	ani = animation.FuncAnimation(fig, update, frames=RUNTIME,fargs=(sc, ax, args.rotation))

	ani.save(args.file, writer='imagemagick', fps=args.fps);
	# ani.save('test_run.mp4', fps=15, extra_args=['-vcodec', 'libx264'])
	# ani.save("demo.avi",codec='avi')
	# ani.save('test_run.mp4', fps=3,extra_args=['-vcodec', 'h264','-pix-fmt', 'yuv420p'])
	
	# plt.show()
	# plt.close()
	print 'Animation saved to {}'.format(args.file)
	return

if __name__ == '__main__':
	main()