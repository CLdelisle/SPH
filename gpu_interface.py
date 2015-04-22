import pycuda.driver as cuda
import pycuda.autoinit
import numpy
from pycuda.compiler import SourceModule
from particle import Particle

class ParticleArrayStruct:
    mem_size = 8 + numpy.intp(0).nbytes
    def __init__(self, array, struct_arr_ptr):
        print "copying data to device"

        self.data = cuda.to_device(array)
        self.shape, self.dtype = array.shape, array.dtype

        cuda.memcpy_htod(int(struct_arr_ptr),
                         numpy.getbuffer(numpy.int32(len(array[0]))))
        cuda.memcpy_htod(int(struct_arr_ptr) + 8,
                         numpy.getbuffer(numpy.intp(int(self.data))))

    def getDataFromDevice(self):
        return cuda.from_device(self.data, self.shape, self.dtype)

def gpuDeviceStats():
  free, total = cuda.mem_get_info()
  print "Global memory occupancy:%f%% free" % (free*100/total)

class ParticleGPUInterface:
  def __init__(self, particles):
    self.mod = SourceModule(self.get_cuda_functions())
    self.cuda_function_cache = {}
    self.number_particles = len(particles)
    print "Allocating memory on the GPU - during sims, you should only see this at the start."
    self.struct_arr = cuda.mem_alloc(2 * ParticleArrayStruct.mem_size)
    particles = [x.flatten() for x in particles]
    self.particles_array = ParticleArrayStruct(numpy.array([particles], numpy.float32), self.struct_arr)

  # collects all the cuda c files
  def get_cuda_functions(self):
    cuda_code = ""
    with open('cuda_lib.c', 'r') as content_file:
        cuda_code += content_file.read()
    with open('cuda_sim.c', 'r') as content_file:
        cuda_code += "\n" + content_file.read()
    with open('cuda_tests.c', 'r') as content_file:
        cuda_code += "\n" + content_file.read()
    return cuda_code

  def get_cuda_function(self, function_name):
    if function_name not in self.cuda_function_cache:
      self.cuda_function_cache[function_name] = self.mod.get_function(function_name)
    return self.cuda_function_cache[function_name]

  def run_cuda_function(self, function_name, params = None):
    if params == None:
      self.get_cuda_function(function_name)(self.struct_arr, block=(32, 1, 1), grid=(self.number_particles / 32, 1))
    else:
      self.get_cuda_function(function_name)(self.struct_arr, params[0], params[1], params[2], block=(32, 1, 1), grid=(self.number_particles / 32, 1))

  def getResultsFromDevice(self):
    return [Particle.unflatten(raw_data) for raw_data in self.particles_array.getDataFromDevice()[0]]