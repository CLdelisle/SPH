import pycuda.driver as cuda
import pycuda.autoinit
import numpy
from pycuda.compiler import SourceModule

from particle import Particle
import time

class DoubleOpStruct:
    mem_size = 8 + numpy.intp(0).nbytes
    def __init__(self, array, struct_array_ptr):
        self.data = cuda.to_device(array)
        self.shape, self.dtype = array.shape, array.dtype
        cuda.memcpy_htod(int(struct_array_ptr), numpy.getbuffer(numpy.int32(array.size)))
        cuda.memcpy_htod(int(struct_array_ptr) + 8, numpy.getbuffer(numpy.intp(int(self.data))))

    def __str__(self):
        return str(cuda.from_device(self.data, self.shape, self.dtype))

#pointer to datasets
datasets = {
  "id": None,
  "mass": None,
  "pos_x": None,
  "pos_y": None,
  "pos_z": None,
  "vel_x": None,
  "vel_y": None,
  "vel_z": None,
  "accel_x": None,
  "accel_y": None,
  "accel_z": None
}

# Iterate through datasets and allocate memory on the device
for dataset_name in datasets:
    datasets[dataset_name] = cuda.mem_alloc(DoubleOpStruct.mem_size)

# Create particles
particles = []
for x in range(0, 4):
    p = Particle(x,x*2,x*3,x*4,x,x,x,x)
    particles.append(p)

# unpack the particles
ids = []
mass = []
rho = []
pressure = []

pos_x = []
pos_y = []
pos_z = []

vel_x = []
vel_y = []
vel_z = []

acc_x = []
acc_y = []
acc_z = []

# populate the arrays with the particle data
for particle in particles:
    ids.append(particle.id)
    mass.append(particle.mass)
    rho.append(particle.rho)
    pressure.append(particle.pressure)

    pos_x.append(particle.pos[0])
    pos_y.append(particle.pos[1])
    pos_z.append(particle.pos[2])

    vel_x.append(particle.vel[0])
    vel_y.append(particle.vel[1])
    vel_z.append(particle.vel[2])

    acc_x.append(particle.acc[0])
    acc_y.append(particle.acc[1])
    acc_z.append(particle.acc[2])

id_array = DoubleOpStruct(numpy.array(ids, dtype=numpy.int16), datasets['id'])
mass_array = DoubleOpStruct(numpy.array(mass, dtype=numpy.float64), datasets['mass'])
pos_x_array = DoubleOpStruct(numpy.array(pos_x, dtype=numpy.float64), datasets['pos_x'])
pos_y_array = DoubleOpStruct(numpy.array(pos_y, dtype=numpy.float64), datasets['pos_y'])
pos_z_array = DoubleOpStruct(numpy.array(pos_z, dtype=numpy.float64), datasets['pos_z'])
vel_x_array = DoubleOpStruct(numpy.array(vel_x, dtype=numpy.float64), datasets['vel_x'])
vel_y_array = DoubleOpStruct(numpy.array(vel_y, dtype=numpy.float64), datasets['vel_y'])
vel_z_array = DoubleOpStruct(numpy.array(vel_z, dtype=numpy.float64), datasets['vel_z'])
accel_x_array = DoubleOpStruct(numpy.array(acc_x, dtype=numpy.float64), datasets['accel_x'])
accel_y_array = DoubleOpStruct(numpy.array(acc_y, dtype=numpy.float64), datasets['accel_y'])
accel_z_array = DoubleOpStruct(numpy.array(acc_z, dtype=numpy.float64), datasets['accel_z'])


print "original arrays"
print id_array
print mass_array
print pos_x_array
mod = SourceModule("""
    struct data_array {
        int array_length;
        int *ptr;
    };

    __global__ void double_array(data_array *id_array, data_array *mass_array, data_array *pos_x) 
    {
        mass_array = mass_array + blockIdx.x;
        for (int idx = threadIdx.x; idx < mass_array->array_length; idx += blockDim.x) 
        {
            int *id_ptr = (int*) id_array->ptr;
            double *mass_ptr = (double*) mass_array->ptr;
            double *pos_x_ptr = (double*) pos_x->ptr;

            mass_ptr[idx] *= 2;
            id_ptr[idx] *= 2;
            pos_x_ptr[idx] *= 2;
        }
    }
    """)
func = mod.get_function("double_array")
func(datasets['id'], datasets['mass'], datasets['pos_x'], block = (len(particles), 1, 1), grid=(1, 1))

print "doubled arrays"
print id_array
print mass_array
print pos_x_array