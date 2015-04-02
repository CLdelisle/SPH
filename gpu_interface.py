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

    def transfer_from_device_to_host(self):
        return cuda.from_device(self.data, self.shape, self.dtype)


class ParticleGPUInterface(object):
  def __init__(self, particles):
    self.particles = particles
    # Define particle attributes and their types
    self.particle_attributes = [
      ParticleAttribute("id", numpy.int32),
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
    self.datasets = {}

    # Iterate through datasets and allocate memory on the device
    for attr in self.particle_attributes:
        self.datasets[attr.name] = {
            "gpu_ptr": cuda.mem_alloc(DoubleOpStruct.mem_size),
            "input": [],
            "results": None, # python numpy array of results in memory
            "result": None # ptr to gpu results
        }

    # populate the arrays with the particle data
    for particle in self.particles:
        self.datasets['id']['input'].append(particle.id)
        self.datasets['mass']['input'].append(particle.mass)
        self.datasets['rho']['input'].append(particle.rho)
        self.datasets['pressure']['input'].append(particle.pressure)

        self.datasets['pos_x']['input'].append(particle.pos[0])
        self.datasets['pos_y']['input'].append(particle.pos[1])
        self.datasets['pos_z']['input'].append(particle.pos[2])

        self.datasets['vel_x']['input'].append(particle.vel[0])
        self.datasets['vel_y']['input'].append(particle.vel[1])
        self.datasets['vel_z']['input'].append(particle.vel[2])

        self.datasets['accel_x']['input'].append(particle.acc[0])
        self.datasets['accel_y']['input'].append(particle.acc[1])
        self.datasets['accel_z']['input'].append(particle.acc[2])

    for attr in self.particle_attributes:
      self.datasets[attr.name]['result'] = DoubleOpStruct(numpy.array(self.datasets[attr.name]['input'], dtype=attr.type), self.datasets[attr.name]['gpu_ptr'])

  # Convert the int, float arrays back to particle objects
  def getResultsFromDevice(self):
    for attr in self.particle_attributes:
      self.datasets[attr.name]['results'] = self.datasets[attr.name]['result'].transfer_from_device_to_host()

    particles = []
    for i in xrange(len(self.datasets['pos_x']['results'])):
        particles.append(
            Particle(
                self.datasets['id']['results'][i],
                self.datasets['mass']['results'][i],
                self.datasets['pos_x']['results'][i],
                self.datasets['pos_y']['results'][i],
                self.datasets['pos_z']['results'][i],
                self.datasets['vel_x']['results'][i],
                self.datasets['vel_y']['results'][i],
                self.datasets['vel_z']['results'][i]
            )
        )

    return particles

  def double_array(self):    
    mod = SourceModule("""
    struct data_array {
        int array_length;
        void *ptr;
    };

    __global__ void double_array(data_array *id_array, data_array *mass_array, data_array *pos_x_array) {
        int idx = threadIdx.x;
        
        int    *id    = (int*)    id_array->ptr;
        double *mass  = (double*) mass_array->ptr;
        double *pos_x = (double*) pos_x_array->ptr;

        id[idx] = (int) (mass[idx] + pos_x[idx]);

    }
    """)
    func = mod.get_function("double_array")
    func(self.datasets['id']['gpu_ptr'], self.datasets['mass']['gpu_ptr'], self.datasets['pos_x']['gpu_ptr'], block = (len(self.particles), 1, 1), grid=(1, 1))