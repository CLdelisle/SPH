
/*
def pressure(p):
  k = 1.0 #this may need to stay hardcoded for our purposes, though could be read in from config file
  gamma = 1.5 #but i'm keeping these constants segregated in this function for now instead of inlining because of this issue
  return (k * (p.rho ** gamma))
*/

__device__ float pressure(Particle* p) {
  //  k = 1.0 #this may need to stay hardcoded for our purposes, though could be read in from config file
  float k = 1.0;

  // gamma = 1.5 #but i'm keeping these constants segregated in this function for now instead of inlining because of this issue
  float gamma = 1.5;
  // return (k * (p.rho ** gamma))
  return (k * powf(p->rho, gamma));
}

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

__device__ float Gaussian_kernel(float* r, float h) {
  // r = np.linalg.norm(r)
  float norm_result = linalg_norm(r);
  // return ( (((1/(np.pi * (h**2)))) ** (3/2) ) * ( np.exp( - ((r**2) / (h**2)) )) )
  return
  (
    powf(
      (1/
        (M_PI * powf(h, 2)
        )
      ), (3.0/2.0))
  )
   *
  powf(M_E,
    ( - ((powf(norm_result, 2) / powf(h, 2)) )));
}


// def cubic_spline_kernel(r, h):
//   # cubic spline function - used if one needs compact support
//   return 0.5 # this is a bullshit placeholder

__device__ float cubic_spline_kernel(float* r, float h) {
  // return 0.5 # this is a bullshit placeholder
  return 0.5;
}

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

__device__ float find_and_execute_kernel(int CHOOSE_KERNEL_CONST, float* r, float h) {
  // if 1 (true) use Gaussian
  // if 0 (false) use spline
  if (CHOOSE_KERNEL_CONST) {
    return Gaussian_kernel(r, h);
  } else {
    return cubic_spline_kernel(r, h);
  }
}

// # Params p,q particles
// def Newtonian_gravity(p,q):
//   # Newton's gravitational constant
//   CONST_G = 6.67384 # * 10^(-11) m^3 kg^-1 s^-2
  
//   '''
//   F = (m_p)a = G(m_p)(m_q)(r)/r^3 -> a = (G * m_q)(r)/(g(r,r)^(3/2)), with g(_,_) the Euclidian inner product
//   Note that this is all in the r-direction vectorially
//   '''

//   r = q.pos - p.pos # separation vector
//   R = np.linalg.norm(r) # magnitude of the separation vector
//   return ((CONST_G * q.mass) / (R**3)) * r

__device__ float* Newtonian_gravity(Particle* p, Particle* q) {
  //   # Newton's gravitational constant
  //   CONST_G = 6.67384 # * 10^(-11) m^3 kg^-1 s^-2
  float CONST_G = 6.67384;
  //   '''
  //   F = (m_p)a = G(m_p)(m_q)(r)/r^3 -> a = (G * m_q)(r)/(g(r,r)^(3/2)), with g(_,_) the Euclidian inner product
  //   Note that this is all in the r-direction vectorially
  //   '''
  //   r = q.pos - p.pos # separation vector
  float r[3];
  vector_difference(r, q->pos, p->pos);

  //   R = np.linalg.norm(r) # magnitude of the separation vector
  float R = linalg_norm(r);
  //   return ((CONST_G * q.mass) / (R**3)) * r

  float scalar = (CONST_G * q->mass) / (powf(R, 3));
  float* result = (float*) malloc(sizeof(float) * 3);
  result[0] = r[0] * scalar;
  result[1] = r[1] * scalar;
  result[2] = r[2] * scalar;
  return result;
}


/**
  for p in particles:
      # preemptively start the Velocity Verlet computation (first half of velocity update part)
      p.vel += (timestep/2.0) * p.acc
      p.temp = p.acc
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
    Particle* p = particle_array->ptr + threadIdx.x;
    // preemptively start the Velocity Verlet computation (first half of velocity update part)
    // p.vel += (timestep/2.0) * p.acc
    for (int i=0; i<3; i++)
      p->vel[i] += (timestep/2.0) * p->acc[i];

    // p.temp = p.acc
    p->temp[0] = p->acc[0];
    p->temp[1] = p->acc[1];
    p->temp[2] = p->acc[2];

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
      float pos_difference[3];
      vector_difference(pos_difference, p->pos, q->pos);
      p->rho = (q->mass * find_and_execute_kernel(CHOOSE_KERNEL_CONST, pos_difference, smooth));

      //   # while we're iterating, add contribution from gravity
      //   if(p.id != q.id):
      if (p->id != q->id) {
        // p.acc += Newtonian_gravity(p,q)
        float* newtonian_gravity_result = Newtonian_gravity(p, q);
        p->acc[0] += newtonian_gravity_result[0];
        p->acc[1] += newtonian_gravity_result[1];
        p->acc[2] += newtonian_gravity_result[2];
        free(&newtonian_gravity_result);

        // # normalize density
        // p.rho = ( p.rho / len(particles) )
        p->rho = p->rho / (float) particle_array->datalen;
        //     p.pressure = pressure(p)
        p->pressure = pressure(p);
      }
    }

}

/*
//second sim loop
for p in particles:
          # acceleration from pressure gradient
          for q in particles:
                  if p.id != q.id:
                    p.acc -= ( q.mass * ((p.pressure / (p.rho ** 2)) + (q.pressure / (q.rho ** 2))) * del_kernel(CHOOSE_KERNEL_CONST, p.pos - q.pos, smooth) ) * (1 / (np.linalg.norm(p.pos - q.pos))) * (p.pos - q.pos)
          # finish velocity update
                                  p.vel += (timestep/2.0) * p.acc

*/

__global__ void second_sim_loop(ParticleArray *particle_array, int timestep, float smooth, int CHOOSE_KERNEL_CONST) {
    Particle* p = particle_array->ptr + threadIdx.x;
}