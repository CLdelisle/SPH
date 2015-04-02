import pycuda.driver as cuda
import pycuda.autoinit
import numpy
from pycuda.compiler import SourceModule

from particle import Particle
import time

class ParticleAttribute:
  def __init__(self, name, type):
    self.name = name
    self.type = type

class DoubleOpStruct:
    mem_size = 8 + numpy.intp(0).nbytes
    def __init__(self, array, struct_array_ptr):
        self.data = cuda.to_device(array)
        self.shape, self.dtype = array.shape, array.dtype
        cuda.memcpy_htod(int(struct_array_ptr), numpy.getbuffer(numpy.int32(array.size)))
        cuda.memcpy_htod(int(struct_array_ptr) + 8, numpy.getbuffer(numpy.intp(int(self.data))))

    def __str__(self):
        return str(cuda.from_device(self.data, self.shape, self.dtype))

# Define particle attributes and their types
particle_attributes = [
  ParticleAttribute("id", numpy.int16),
  ParticleAttribute("mass", numpy.float64),
  ParticleAttribute("rho", numpy.float64),
  ParticleAttribute("pressure", numpy.float64),
  ParticleAttribute("pos_x", numpy.float64),
  ParticleAttribute("pos_y", numpy.float64),
  ParticleAttribute("pos_z", numpy.float64),
  ParticleAttribute("vel_x", numpy.float64),
  ParticleAttribute("vel_y", numpy.float64),
  ParticleAttribute("vel_z", numpy.float64),
  ParticleAttribute("accel_x", numpy.float64),
  ParticleAttribute("accel_y", numpy.float64),
  ParticleAttribute("accel_z", numpy.float64)
]

#pointer to datasets(id, mass, accel_z, etc.)
datasets = {}

# Iterate through datasets and allocate memory on the device
for attr in particle_attributes:
    datasets[attr.name] = {
        "gpu_ptr": cuda.mem_alloc(DoubleOpStruct.mem_size),
        "input": [],
        "results": None
    }

# Create particles
particles = []
for x in range(0, 4):
    p = Particle(x,x*2,x*3,x*4,x,x,x,x)
    particles.append(p)

# populate the arrays with the particle data
for particle in particles:
    datasets['id']['input'].append(particle.id)
    datasets['mass']['input'].append(particle.mass)
    datasets['rho']['input'].append(particle.rho)
    datasets['pressure']['input'].append(particle.pressure)

    datasets['pos_x']['input'].append(particle.pos[0])
    datasets['pos_y']['input'].append(particle.pos[1])
    datasets['pos_z']['input'].append(particle.pos[2])

    datasets['vel_x']['input'].append(particle.vel[0])
    datasets['vel_y']['input'].append(particle.vel[1])
    datasets['vel_z']['input'].append(particle.vel[2])

    datasets['accel_x']['input'].append(particle.acc[0])
    datasets['accel_y']['input'].append(particle.acc[1])
    datasets['accel_z']['input'].append(particle.acc[2])

for attr in particle_attributes:
  datasets[attr.name]['result'] = DoubleOpStruct(numpy.array(datasets[attr.name]['input'], dtype=attr.type), datasets[attr.name]['gpu_ptr'])

print "original arrays"
print datasets['id']['result']
print datasets['mass']['result']
print datasets['pos_x']['result']
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

            mass_ptr[idx] *= 10;
            id_ptr[idx] *= 100;
            pos_x_ptr[idx] *= 1000;
        }
    }
    """)
func = mod.get_function("double_array")
func(datasets['id']['gpu_ptr'], datasets['mass']['gpu_ptr'], datasets['pos_x']['gpu_ptr'], block = (len(particles), 1, 1), grid=(1, 1))

print "doubled arrays"
print datasets['id']['result']
print datasets['mass']['result']
print datasets['pos_x']['result']