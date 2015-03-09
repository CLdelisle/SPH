from mpl_toolkits.mplot3d import axes3d
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from outputFiles import sortedFileNames

prefix = raw_input('Output file prefix(defaults to \'output\'): ')
prefix = prefix or 'example'

files = sortedFileNames(prefix)

t = 1 #global "time" counter
RUNTIME = len(files) #Number of frames to render

curr_x = []
curr_y = []
curr_z = []

data = []
# data file format: Particle ID, X-coord, Y-coord, Z-coord, X-Velocity, Y-Velocity, Z-Velocity
for i, filename in enumerate(files):
	data.append(np.genfromtxt(filename, delimiter=',', names=['pid', 'mass', 'x', 'y', 'z', 'vx', 'vy', 'vz']))

def update(num,sc,ax):
	ax.cla();
	global N
	global t
	global data
	global curr_x
	global curr_y
	global curr_z

	curr_x = []
	curr_y = []
	curr_z = []

	for k in range(len(files)):
		for i in range (0, len(data[0]['x'])):
			curr_x.append(data[k]['x'][i])
			curr_y.append(data[k]['y'][i])
			curr_z.append(data[k]['z'][i])

	t = t+1
	print "t=".format(t)
	ax.autoscale(False)
	sc = ax.scatter(curr_x, curr_y, curr_z, c='m', marker='o')
	return sc

def main():
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	curr_x = []
	curr_y = []
	curr_z = []
	for i in range (0, len(data[0]['x'])):
		curr_x.append(data[0]['x'][i])
		curr_y.append(data[0]['y'][i])
		curr_z.append(data[0]['z'][i])
		
		
	ax.set_xlim3d(-700,700);
	ax.set_ylim3d(-700,700);
	ax.set_zlim3d(-700,700);

	sc=ax.scatter(curr_x, curr_y, curr_z, c='r', marker='o')

	#ax1.plot(data['t'],data['x'],color='r',label='position')
	#ax.scatter(curr_x, curr_y, curr_z, c='r', marker='o')

	ani = animation.FuncAnimation(fig, update, frames=RUNTIME,fargs=(sc,ax),interval=100)
	
	ani.save('test_run.gif', writer='imagemagick', fps=15);
	#ani.save('test_run.mp4', fps=15, extra_args=['-vcodec', 'libx264'])
	#ani.save("demo.avi",codec='avi')
	#ani.save('test_run.mp4', fps=30,extra_args=['-vcodec', 'h264','-pix-fmt', 'yuv420p'])
	
	#plt.show()
	#plt.close()
	return

if __name__ == '__main__':
	main()