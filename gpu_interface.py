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
        """
        numpy.getbuffer() needed due to lack of new-style buffer interface for
        scalar numpy arrays as of numpy version 1.9.1

        see: https://github.com/inducer/pycuda/pull/60
        """
        cuda.memcpy_htod(int(struct_arr_ptr),
                         numpy.getbuffer(numpy.int32(array.size)))
        cuda.memcpy_htod(int(struct_arr_ptr) + 8,
                         numpy.getbuffer(numpy.intp(int(self.data))))

    def getDataFromDevice(self):
        print "getDataFromDevice()"
        return cuda.from_device(self.data, self.shape, self.dtype)

class ParticleGPUInterface:
  def __init__(self, particles):
    self.mod = SourceModule(self.get_cuda_functions())
    self.cuda_function_cache = {}

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

  # gpu_particles.first_sim_loop(timestep, smooth, CHOOSE_KERNEL_CONST)
  def sim_loop(self, function_name, timestep, smooth, CHOOSE_KERNEL_CONST):
    if function_name in self.cuda_function_cache:
      self.cuda_function_cache[function_name](self.struct_arr, numpy.int32(timestep), numpy.float32(smooth), numpy.int32(CHOOSE_KERNEL_CONST), block=(32, 1, 1), grid=(1, 1))
    else:
      warnings.warn("Cuda function {} not found in cache.".format(function_name), Warning)
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
    return [Particle.unflatten(raw_data) for raw_data in self.particles_array.getDataFromDevice()[0]]