// MUST match the particle.flatten() format
//   return [self.id, self.mass, self.pos[0], self.pos[1], self.pos[2], self.vel[0], self.vel[1], self.vel[2], self.acc[0], self.acc[1], self.acc[2], self.rho, self.pressure]

struct Particle {
  float id; //must be same type as the other properties
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

float find_kernel(int CHOOSE_KERNEL_CONST, float* r, float h) {
  return 0;
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
    Particle* particle = particle_array->ptr + blockIdx.x;
    // preemptively start the Velocity Verlet computation (first half of velocity update part)
    // p.vel += (timestep/2.0) * p.acc
    particle->vel_x += (timestep/2.0) * particle->acc_x;
    particle->vel_y += (timestep/2.0) * particle->acc_y;
    particle->vel_z += (timestep/2.0) * particle->acc_z;

    // temp = p.acc
    float temp[3] = {particle->acc_x, particle->acc_y, particle->acc_z};

    // p.acc = 0.0
    particle->acc_x = 0;
    particle->acc_y = 0;
    particle->acc_z = 0;

    // p.rho = 0.0
    particle->rho = 0;
    // p.pressure = 0.0
    particle->pressure = 0;

    // particle.
    //     Particle *particle = a->ptr;
    //     particle[idx].mass = particle[idx].pos_x + particle[idx].pos_y;
    // }
}