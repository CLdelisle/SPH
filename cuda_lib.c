#include <math.h>
//This file contains general helper functions for the cuda implementation


// MUST match the particle.flatten() format
//   return [self.id, self.mass, self.pos[0], self.pos[1], self.pos[2], self.vel[0], self.vel[1], self.vel[2], self.acc[0], self.acc[1], self.acc[2], self.rho, self.pressure]

struct Particle {
  float id, //must be same type as the other propertie
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
