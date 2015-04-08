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
}

__global__ void vector_difference_test(ParticleArray *particle_array) {
    Particle* p = particle_array->ptr + threadIdx.x;
    vector_difference(p->acc, p->pos, p->vel);
}