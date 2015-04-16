import pycuda.driver as cuda
import pycuda.autoinit
import numpy
from pycuda.compiler import SourceModule
from particle import Particle
import warnings

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
        print "getDataFromDevice() start"
        a = cuda.from_device(self.data, self.shape, self.dtype)
        print "getDataFromDevice() end"
        return a

def gpuDeviceStats():
  free, total = cuda.mem_get_info()
  print "Global memory occupancy:%f%% free" % (free*100/total)

class ParticleGPUInterface:
  def __init__(self, particles):
    self.mod = SourceModule(self.get_cuda_functions())
    self.cuda_function_cache = {}
    self.number_particles = len(particles)

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
    return cuda_code

  def sim_loop(self, function_name, timestep, smooth, CHOOSE_KERNEL_CONST):
    gpuDeviceStats()
    if function_name in self.cuda_function_cache:
      self.cuda_function_cache[function_name](self.struct_arr, numpy.int32(timestep), numpy.float32(smooth), numpy.int32(CHOOSE_KERNEL_CONST), block=(self.number_particles, 1, 1), grid=(1, 1))
    else:
      print "Cuda function {} not found in cache.".format(function_name)
      self.cuda_function_cache[function_name] = self.mod.get_function(function_name)

  # Runs cuda tests
  def cudaTests(self, test_name, number_particles):
    cuda_code = self.get_cuda_functions()
    # Append the test functions to the sim and lib code
    with open('cuda_tests.c', 'r') as content_file:
        cuda_code += "\n" + content_file.read()
    mod = SourceModule(cuda_code)
    func = mod.get_function(test_name)
    func(self.struct_arr, block=(number_particles, 1, 1), grid=(1, 1))

  def getResultsFromDevice(self):
    print "getResultsFromDevice() start"
    a = [Particle.unflatten(raw_data) for raw_data in self.particles_array.getDataFromDevice()[0]]
    print "getResultsFromDevice() end"
    return a