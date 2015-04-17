#include <math.h>
//This file contains general helper functions for the cuda implementation


// MUST match the particle.flatten() format
//   return [self.id, self.mass, self.pos[0], self.pos[1], self.pos[2], self.vel[0], self.vel[1], self.vel[2], self.acc[0], self.acc[1], self.acc[2], self.rho, self.pressure]

struct Particle {
  float id, //must be same type as the other properties
        mass,
        pos[3],
        vel[3],
        acc[3],
        rho,
        pressure,
        temp[3];
};


struct ParticleArray {
    int datalen, __padding; // so 64-bit ptrs can be aligned
    Particle *ptr;
};

__device__ int ids_are_equal(float id1, float id2) {
  return (fabsf(id1) - fabsf(id2) <= 0.00001);
}

// subtract vectors
// a - b, result stored in first param
__device__ void vector_difference(float* result, float* a, float* b) {
  for (int i=0; i<3; i++)
    result[i] = a[i] - b[i];
}

//np.linalg.norm
//http://thisthread.blogspot.com/2012/03/frobenius-norm.html

__device__ float linalg_norm(float* matrix) {
    return sqrt(matrix[0] * matrix[0] + matrix[1] * matrix[1] + matrix[2] * matrix[2]);
}

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

// def del_Gaussian(r, h):
//   # derivative of Gaussian kernel
//   r1 = np.linalg.norm(r)
//   return ( ((-2 * r1) / (h**2)) * Gaussian_kernel(r, h))

__device__ float del_Gaussian(float* r, float h) {
  float r1 = linalg_norm(r);
  return (((-2 * r1) / (powf(h, 2))) * Gaussian_kernel(r, h));
}


// def del_cubic_spline(r, h):
//   # derivative of cubic spline
//   return 0.5 # this is a bullshit placeholder

__device__ float del_cubic_spline(float* r, float h) {
  return 0.5;
}

// def del_kernel(x, r, h):
//   # if 1 (true) use Gaussian
//   # if 0 (false) use spline
//   if(x):
//     return del_Gaussian(r, h)
//   else:
//     return del_cubic_spline(r, h)

__device__ float del_kernel(int x, float* r, float h) {
  if (x) {
    return del_Gaussian(r, h);
  } else {
    return del_cubic_spline(r, h);
  }
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