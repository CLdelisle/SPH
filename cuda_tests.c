#include <stdio.h>

__global__ void increment_particle_properties(ParticleArray *particle_array) {
    Particle* p = particle_array->ptr + threadIdx.x;
    p->id       += 1;
    p->mass     += 1;
    p->pos[0]   += 1;
    p->pos[1]   += 1;
    p->pos[2]   += 1;
    p->vel[0]   += 1;
    p->vel[1]   += 1;
    p->vel[2]   += 1;
    p->acc[0]   += 1;
    p->acc[1]   += 1;
    p->acc[2]   += 1;
    p->rho      += 1;
    p->pressure += 1;
    p->temp[0]   += 1;
    p->temp[1]   += 1;
    p->temp[2]   += 1;
}

__global__ void increment_particle_properties_on_multiple_particles(ParticleArray *particle_array) {
    Particle* p = particle_array->ptr + threadIdx.x;
    p->id       += 1;
    p->mass     += 1;
    p->pos[0]   += 1;
    p->pos[1]   += 1;
    p->pos[2]   += 1;
    p->vel[0]   += 1;
    p->vel[1]   += 1;
    p->vel[2]   += 1;
    p->acc[0]   += 1;
    p->acc[1]   += 1;
    p->acc[2]   += 1;
    p->rho      += 1;
    p->pressure += 1;
    p->temp[0]   += 1;
    p->temp[1]   += 1;
    p->temp[2]   += 1;
}

__global__ void vector_difference_test(ParticleArray *particle_array) {
    Particle* p = particle_array->ptr + threadIdx.x;
    vector_difference(p->acc, p->pos, p->vel);
}

__global__ void particle_pressure_test(ParticleArray *particle_array) {
    Particle* p = particle_array->ptr + threadIdx.x;
    p->pressure = pressure(p);
}

__global__ void gaussian_kernel_test(ParticleArray *particle_array) {
    Particle* p = particle_array->ptr + threadIdx.x;
    //I know these values don't normally seed this function, but I'm just throwing a random vector and float at it.
    p->pressure = Gaussian_kernel(p->acc, p->rho);
}


//confirm that the number of particles is correctly received by the GPU
__global__ void number_of_particles_test(ParticleArray *particle_array) {
    Particle* p = particle_array->ptr + threadIdx.x;
    p->id = particle_array->datalen;
}