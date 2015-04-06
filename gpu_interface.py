import pycuda.driver as cuda
import pycuda.autoinit
import numpy
from pycuda.compiler import SourceModule
from particle import Particle

class ParticleArrayStruct:
    mem_size = 8 + numpy.intp(0).nbytes
    def __init__(self, array, struct_arr_ptr):
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
        return cuda.from_device(self.data, self.shape, self.dtype)

class ParticleGPUInterface:
  def __init__(self, particles):
    self.struct_arr = cuda.mem_alloc(2 * ParticleArrayStruct.mem_size)
    particles = [x.flatten() for x in particles]
    self.particles_array = ParticleArrayStruct(numpy.array([particles], numpy.float32), self.struct_arr)

  # collects all the cuda c files with the import order determined by their sorted file names
  def get_cuda_functions(self):
    with open('cuda.c', 'r') as content_file:
        return content_file.read()

  # gpu_particles.first_sim_loop(timestep, smooth, CHOOSE_KERNEL_CONST)
  def first_sim_loop(self, timestep, smooth, CHOOSE_KERNEL_CONST):
    cuda_code = self.get_cuda_functions()
    mod = SourceModule(cuda_code)
    func = mod.get_function("first_sim_loop")
    func(self.struct_arr, numpy.int32(timestep), numpy.float32(smooth), numpy.int32(CHOOSE_KERNEL_CONST), block=(32, 1, 1), grid=(2, 1))

  def getResultsFromDevice(self):
    return [Particle.unflatten(raw_data) for raw_data in self.particles_array.getDataFromDevice()[0]]