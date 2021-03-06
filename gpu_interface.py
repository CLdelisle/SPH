import pycuda.driver as cuda
import pycuda.autoinit
import numpy
from pycuda.compiler import SourceModule
from particle import Particle

# Structure that holds pointers to the GPU object
# first 8 bytes are the length of the particle array
# next n bytes points to the array of particles
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

    # copies bytes from device to host
    def getDataFromDevice(self):
        return cuda.from_device(self.data, self.shape, self.dtype)

# debugging function that prints that amount of GPU global memory that is free.
# Not called from the sim or tests
def gpuDeviceStats():
  free, total = cuda.mem_get_info()
  print "Global memory occupancy:%f%% free" % (free*100/total)

# main API for transferring particles to and from the GPU
class ParticleGPUInterface:
  # Initialize with the array of particles
  def __init__(self, particles):
    self.mod = SourceModule(self.get_cuda_code())
    self.cuda_function_cache = {}
    self.number_particles = len(particles)
    print "Allocating memory on the GPU - during sims, you should only see this at the start."
    self.struct_arr = cuda.mem_alloc(2 * ParticleArrayStruct.mem_size)

    # Flatten the particle objects to an array of floats to be transferred
    particles = [x.flatten() for x in particles]
    self.particles_array = ParticleArrayStruct(numpy.array([particles], numpy.float32), self.struct_arr)

  # collects all the cuda c files
  # currently hardcoded for 3 files (library, simulation, and test)
  # Reminder - this is a SLOW function since it's file i/o. Only call this once!
  def get_cuda_code(self):
    cuda_code = ""
    with open('cuda_lib.cu', 'r') as content_file:
        cuda_code += content_file.read()
    with open('cuda_sim.cu', 'r') as content_file:
        cuda_code += "\n" + content_file.read()
    with open('cuda_tests.cu', 'r') as content_file:
        cuda_code += "\n" + content_file.read()
    return cuda_code

  # returns a cuda function given its name.
  # cuda methods are automatically cached
  def get_cuda_function(self, function_name):
    if function_name not in self.cuda_function_cache:
      self.cuda_function_cache[function_name] = self.mod.get_function(function_name)
    return self.cuda_function_cache[function_name]

  # Call this method to run a cuda function.
  # Supply its name and optional parameters
  # Right now, parameters are hardcoded for the sim.  They should be variable in the future.
  def run_cuda_function(self, function_name, params = None):
    blocks = (32, 1, 1)

    num_blocks = self.number_particles / 32
    if self.number_particles % 32 != 0:
      num_blocks += 1

    grid = (num_blocks, 1)
    if params == None:
      self.get_cuda_function(function_name)(self.struct_arr, block=blocks, grid=grid)
    else:
      self.get_cuda_function(function_name)(self.struct_arr, params[0], params[1], params[2], block=blocks, grid=grid)

  # transfers float array fromt the GPU and converts the floats to particles
  def getResultsFromDevice(self):
    return [Particle.unflatten(raw_data) for raw_data in self.particles_array.getDataFromDevice()[0]]