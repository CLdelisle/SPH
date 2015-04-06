// MUST match the particle.flatten() format
//   return [self.id, self.mass, self.pos[0], self.pos[1], self.pos[2], self.vel[0], self.vel[1], self.vel[2], self.acc[0], self.acc[1], self.acc[2], self.rho, self.pressure]

struct Particle {
  float id; //must be same type as the other properties
  float mass;

  float pos[3];

  float vel[3];

  float acc[3];

  float rho;
  float pressure;
};


struct ParticleArray {
    int datalen, __padding; // so 64-bit ptrs can be aligned
    Particle *ptr;
};

//np.linalg.norm

// subtract vectors

/*
params
r - position vector difference (float[])
h - smooth (float)
x - CHOOSE_KERNEL_CONST (int)

def find_kernel(x, r, h):
  # if 1 (true) use Gaussian
  # if 0 (false) use spline
  if(x):
    return Gaussian_kernel(r, h)
  else:
    return cubic_spline_kernel(r, h)
*/

/*
params
r - position vector difference
h - smooth (float)

returns float
def Gaussian_kernel(r, h):
  # Gaussian function
  r = np.linalg.norm(r)
  return ( (((1/(np.pi * (h**2)))) ** (3/2) ) * ( np.exp( - ((r**2) / (h**2)) )) )  
*/

__device__ float find_and_execute_kernel(int CHOOSE_KERNEL_CONST, float* r, float h) {
  return 1;
}

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

// a - b
__device__ float* vector_difference(float a[3], float b[3]) {
  for (int i=0; i<3; i++) {
    a[i] = a[i] - b[i];
  }
  return a;
}

/**
  for p in particles:
      # preemptively start the Velocity Verlet computation (first half of velocity update part)
      p.vel += (timestep/2.0) * p.acc
      temp = p.acc
      p.acc = 0.0
      p.rho = 0.0
      p.pressure = 0.0
      #get density
      for q in particles:
        p.rho += ( q.mass * (find_kernel(CHOOSE_KERNEL_CONST, p.pos - q.pos, smooth)) )
        # while we're iterating, add contribution from gravity
        if(p.id != q.id):
          p.acc += Newtonian_gravity(p,q)
      # normalize density
      p.rho = ( p.rho / len(particles) )
          p.pressure = pressure(p)
**/


__global__ void first_sim_loop(ParticleArray *particle_array, int timestep, float smooth, int CHOOSE_KERNEL_CONST) {
    // for p in particles
    Particle* p = particle_array->ptr + blockIdx.x;
    // preemptively start the Velocity Verlet computation (first half of velocity update part)
    // p.vel += (timestep/2.0) * p.acc
    for (int i=0; i<3; i++)
      p->vel[i] += (timestep/2.0) * p->acc[i];

    // temp = p.acc
    float temp[3] = {p->acc[0], p->acc[1], p->acc[2]};

    // p.acc = 0.0
    p->acc[0] = 0;
    p->acc[1] = 0;
    p->acc[2] = 0;

    // p.rho = 0.0
    p->rho = 0;
    // p.pressure = 0.0
    p->pressure = 0;

    // #get density
    // for q in particles:
    for (int i=0; i<particle_array->datalen; i++) {
      Particle* q = particle_array->ptr + i;

      //   p.rho += ( q.mass * (find_kernel(CHOOSE_KERNEL_CONST, p.pos - q.pos, smooth)) )
      p->rho = (q->mass * find_and_execute_kernel(CHOOSE_KERNEL_CONST, vector_difference(p->pos, q->pos), smooth));

      //   # while we're iterating, add contribution from gravity
      //   if(p.id != q.id):
      //     p.acc += Newtonian_gravity(p,q)
      // # normalize density
      // p.rho = ( p.rho / len(particles) )
      //     p.pressure = pressure(p)
    }

}