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
dataset_ptrs = {
  "id_array_ptr": None,
  "mass_array_ptr": None,
  "pos_x_ptr": None,
  "pos_y_ptr": None,
  "pos_z_ptr": None,
  "vel_x_ptr": None,
  "vel_y_ptr": None,
  "vel_z_ptr": None,
  "accel_x_ptr": None,
  "accel_y_ptr": None,
  "accel_z_ptr": None
}

for dataset_name in dataset_ptrs:
    dataset_ptrs[dataset_name] = cuda.mem_alloc(DoubleOpStruct.mem_size)

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

id_array = DoubleOpStruct(numpy.array(ids, dtype=numpy.int16), dataset_ptrs['id_array_ptr'])
mass_array = DoubleOpStruct(numpy.array(mass, dtype=numpy.float64), dataset_ptrs['mass_array_ptr'])
pos_x_array = DoubleOpStruct(numpy.array(pos_x, dtype=numpy.float64), dataset_ptrs['pos_x_ptr'])
pos_y_array = DoubleOpStruct(numpy.array(pos_y, dtype=numpy.float64), dataset_ptrs['pos_y_ptr'])
pos_z_array = DoubleOpStruct(numpy.array(pos_z, dtype=numpy.float64), dataset_ptrs['pos_z_ptr'])
vel_x_array = DoubleOpStruct(numpy.array(vel_x, dtype=numpy.float64), dataset_ptrs['vel_x_ptr'])
vel_y_array = DoubleOpStruct(numpy.array(vel_y, dtype=numpy.float64), dataset_ptrs['vel_y_ptr'])
vel_z_array = DoubleOpStruct(numpy.array(vel_z, dtype=numpy.float64), dataset_ptrs['vel_z_ptr'])
accel_x_array = DoubleOpStruct(numpy.array(acc_x, dtype=numpy.float64), dataset_ptrs['accel_x_ptr'])
accel_y_array = DoubleOpStruct(numpy.array(acc_y, dtype=numpy.float64), dataset_ptrs['accel_y_ptr'])
accel_z_array = DoubleOpStruct(numpy.array(acc_z, dtype=numpy.float64), dataset_ptrs['accel_z_ptr'])


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
func(dataset_ptrs['id_array_ptr'], dataset_ptrs['mass_array_ptr'], dataset_ptrs['pos_x_ptr'], block = (len(particles), 1, 1), grid=(1, 1))

print "doubled arrays"
print id_array
print mass_array
print pos_x_array