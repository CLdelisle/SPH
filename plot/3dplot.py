from mpl_toolkits.mplot3d import axes3d
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation

N = 60 #NUM_BODIES - READ FROM HEADER
t = 1 #global "time" counter
RUNTIME = 200 #Number of frames to render

curr_x = []
curr_y = []
curr_z = []
data = np.genfromtxt('output.csv', delimiter=',', names=['x','y','z'])

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
	for i in range (0,N):
		curr_x.append(data['x'][i+(N*t)])
		curr_y.append(data['y'][i+(N*t)])
		curr_z.append(data['z'][i+(N*t)])
		#print (i, data['x'][i+(N*t)], data['y'][i+(N*t)], data['z'][i+(N*t)])
	t = t+1
	print t
	ax.autoscale(False)
	sc = ax.scatter(curr_x, curr_y, curr_z, c='m', marker='o')
	return sc

def main():
	fig = plt.figure()
	ax = fig.add_subplot(111,projection='3d')

	curr_x = []
	curr_y = []
	curr_z = []
	for i in range (0,N):
		curr_x.append(data['x'][i])
		curr_y.append(data['y'][i])
		curr_z.append(data['z'][i])
		
		
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
