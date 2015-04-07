#include <stdio.h>

__global__ void cuda_tests(ParticleArray *particle_array, int timestep, float smooth, int CHOOSE_KERNEL_CONST) {
    printf("Running the tests on particle %d %d %d.\n", threadIdx.x);
    Particle* p = particle_array->ptr + threadIdx.x;

}