#include <stdio.h>
#include <stdlib.h>
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

__device__ void first_sim_loop(ParticleArray *particle_array, int timestep, float smooth, int CHOOSE_KERNEL_CONST) {
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
      // printf("pos_difference[1]: %f\n", pos_difference[1]);
      // printf("pos_difference[1]: %f\n", pos_difference[1]);
      vector_difference(pos_difference, p->pos, q->pos);
      
      p->rho += (q->mass * find_and_execute_kernel(CHOOSE_KERNEL_CONST, pos_difference, smooth));
      
      //   # while we're iterating, add contribution from gravity
      //   if(p.id != q.id):
      if (!ids_are_equal(p->id, q->id)) {
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

__device__ void second_sim_loop(ParticleArray *particle_array, int timestep, float smooth, int CHOOSE_KERNEL_CONST) {
    Particle* p = particle_array->ptr + threadIdx.x;

    // for q in particles:
    for (int i=0; i<particle_array->datalen; i++) {
      Particle* q = particle_array->ptr + i;
      // if p.id != q.id:
        if (!ids_are_equal(p->id, q->id)) {
          float pos_difference[3];
          vector_difference(pos_difference, p->pos, q->pos);
          float a = ((p->pressure / powf(p->rho, 2)) + (q->pressure / powf(q->rho, 2)));
          float b = (q->mass * a * del_kernel(CHOOSE_KERNEL_CONST, pos_difference, smooth)) * (1 / linalg_norm(pos_difference));
          p->acc[0] -= b * pos_difference[0];
          p->acc[1] -= b * pos_difference[1];
          p->acc[2] -= b * pos_difference[2];
        }
    }
}

/*
  third and final sim loop
  for p in particles:
    # perform position update
    p.pos += timestep * (p.vel + (timestep/2.0)*p.temp)
*/
__device__ void third_sim_loop(ParticleArray *particle_array, int timestep, float smooth, int CHOOSE_KERNEL_CONST) {
  Particle* p = particle_array->ptr + threadIdx.x;
    // p.pos += timestep * (p.vel + (timestep/2.0)*p.temp)
  for (int i=0; i<3; i++) {
    p->pos[i] += timestep * (p->vel[i] + (timestep/2.0) * p->temp[i]);
  }
}

// Runs all sim loops
__global__ void run_simulation_loops(ParticleArray *particle_array, int timestep, float smooth, int CHOOSE_KERNEL_CONST) {
  if (threadIdx.x < particle_array->datalen) {
    first_sim_loop(particle_array, timestep, smooth, CHOOSE_KERNEL_CONST);
    second_sim_loop(particle_array, timestep, smooth, CHOOSE_KERNEL_CONST);
    third_sim_loop(particle_array, timestep, smooth, CHOOSE_KERNEL_CONST);

    Particle* p = particle_array->ptr + threadIdx.x;
    printf("particle %d: pos[0]=%f pressure=%f\n", (int) p->id, p->pos[0], p->pressure);
  } else {
    printf("not running on index %d\n", threadIdx.x);
  }
}