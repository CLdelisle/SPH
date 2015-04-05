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

  # gpu_particles.first_sim_loop(timestep, smooth, CHOOSE_KERNEL_CONST)
  def first_sim_loop(self, timestep, smooth, CHOOSE_KERNEL_CONST):
    mod = SourceModule("""
        // MUST match the particle.flatten() format
        //   return [self.id, self.mass, self.pos[0], self.pos[1], self.pos[2], self.vel[0], self.vel[1], self.vel[2], self.acc[0], self.acc[1], self.acc[2], self.rho, self.pressure]

        struct Particle {
          float id; //must be same type
          float mass;

          float pos_x;
          float pos_y;
          float pos_z;

          float vel_x;
          float vel_y;
          float vel_z;

          float acc_x;
          float acc_y;
          float acc_z;

          float rho;
          float pressure;
        };

        struct ParticleArray {
            int datalen, __padding; // so 64-bit ptrs can be aligned
            Particle *ptr;
        };

        //        float Newtonian_gravity(Particle p, Particle q) {
        //          // Newton's gravitational constant
        //          CONST_G = 6.67384 # * 10^(-11) m^3 kg^-1 s^-2
        //
        //          /**
        //            F = (m_p)a = G(m_p)(m_q)(r)/r^3 -> a = (G * m_q)(r)/(g(r,r)^(3/2)), with g(_,_) the Euclidian inner product
        //            Note that this is all in the r-direction vectorially
        //          **/
        //
        //          float r = q.pos - p.pos # separation vector
        //          float R = np.linalg.norm(r) # magnitude of the separation vector
        //          return ((CONST_G * q.mass) / (R**3)) * r
        //        }
        //
        //
        __global__ void first_sim_loop(ParticleArray *a, int timestep, float smooth, int CHOOSE_KERNEL_CONST) {
            a = a + blockIdx.x;
            for (int idx = threadIdx.x; idx < a->datalen; idx += blockDim.x) {
                Particle *particle = a->ptr;
                particle[idx].mass = particle[idx].pos_x + particle[idx].pos_y;
            }
        }
        """)
    func = mod.get_function("first_sim_loop")
    func(self.struct_arr, numpy.int32(timestep), numpy.float32(smooth), numpy.int32(CHOOSE_KERNEL_CONST), block=(32, 1, 1), grid=(2, 1))

  def getResultsFromDevice(self):
    return [Particle.unflatten(raw_data) for raw_data in self.particles_array.getDataFromDevice()[0]]