# prepared invocations and structures -----------------------------------------

import pycuda.driver as cuda
import pycuda.autoinit
import numpy
from pycuda.compiler import SourceModule
from particle import Particle

class DoubleOpStruct:
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

    def __str__(self):
        return str(cuda.from_device(self.data, self.shape, self.dtype))

struct_arr = cuda.mem_alloc(2 * DoubleOpStruct.mem_size)
do2_ptr = int(struct_arr) + DoubleOpStruct.mem_size

# Generate test particles
particles = [
  Particle(2,2.35,2.78,2,2,2,2,2),
  Particle(3,3.89,3.105,3,3,3,3,3),
  Particle(4,4.789,4.456,4.197,4,4,4,4),
]

particles = bar = [x.flatten() for x in particles]

particles_array = DoubleOpStruct(numpy.array([particles], numpy.float32), struct_arr)

print "original arrays"
print particles_array

mod = SourceModule("""
    struct Particle {
      float id; //must be same type
      float mass;
      float pos_x;
      float pos_y;
      float pos_z;
    };

    struct DoubleOperation {
        int datalen, __padding; // so 64-bit ptrs can be aligned
        Particle *ptr;
    };

    __global__ void double_array(DoubleOperation *a)
    {
        a = a + blockIdx.x;
        for (int idx = threadIdx.x; idx < a->datalen; idx += blockDim.x)
        {
            Particle *a_ptr = a->ptr;
            a_ptr[idx].mass *= 2;
        }
    }
    """)
func = mod.get_function("double_array")
func(struct_arr, block=(32, 1, 1), grid=(2, 1))

print "doubled arrays"
print particles_array