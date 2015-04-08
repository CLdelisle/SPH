from particle import Particle
from gpu_interface import ParticleGPUInterface
import random
import numpy as np

required_decimal_accuray = 2

def randomFloat():
	return random.uniform(1, 1e3)

def generateTestParticle():
	return Particle(random.randint(1, 10000), randomFloat(), randomFloat(), randomFloat(), randomFloat(), randomFloat(), randomFloat(), randomFloat(),
		acc=[randomFloat(), randomFloat(), randomFloat()], rho=randomFloat(), pre=randomFloat())

def generateParticles(n):
	return [generateTestParticle() for i in xrange(n)]


# Test 1 - increment all particle property values by 1
particles_test_data = generateParticles(1)
gpu_particles = ParticleGPUInterface(particles_test_data)
gpu_particles.cudaTests("increment_particle_properties", len(particles_test_data))
updated_particle = gpu_particles.getResultsFromDevice()[0]


np.testing.assert_almost_equal(updated_particle.id, particles_test_data[0].id + 1, required_decimal_accuray)
np.testing.assert_almost_equal(updated_particle.mass, particles_test_data[0].mass + 1, required_decimal_accuray)
np.testing.assert_almost_equal(updated_particle.pos, particles_test_data[0].pos + 1, required_decimal_accuray)
np.testing.assert_almost_equal(updated_particle.vel, particles_test_data[0].vel + 1, required_decimal_accuray)
np.testing.assert_almost_equal(updated_particle.acc, particles_test_data[0].acc + 1, required_decimal_accuray)
np.testing.assert_almost_equal(updated_particle.pressure, particles_test_data[0].pressure + 1, required_decimal_accuray)
np.testing.assert_almost_equal(updated_particle.rho, particles_test_data[0].rho + 1, required_decimal_accuray)


# Test 2 - Same as test 1, but apply +1 to all properties on multiple particles. Tests thread access index
particles_test_data = generateParticles(5)

gpu_particles = ParticleGPUInterface(particles_test_data)
gpu_particles.cudaTests("increment_particle_properties_on_multiple_particles", len(particles_test_data))
updated_particles = gpu_particles.getResultsFromDevice()

for idx in xrange(len(particles_test_data)):
	np.testing.assert_almost_equal(updated_particles[idx].id, particles_test_data[idx].id + 1, required_decimal_accuray)
	np.testing.assert_almost_equal(updated_particles[idx].mass, particles_test_data[idx].mass + 1, required_decimal_accuray)
	np.testing.assert_almost_equal(updated_particles[idx].pressure, particles_test_data[idx].pressure + 1, required_decimal_accuray)
	np.testing.assert_almost_equal(updated_particles[idx].rho, particles_test_data[idx].rho + 1, required_decimal_accuray)
	np.testing.assert_almost_equal(updated_particles[idx].pos, particles_test_data[idx].pos + 1, required_decimal_accuray)
	np.testing.assert_almost_equal(updated_particles[idx].vel, particles_test_data[idx].vel + 1, required_decimal_accuray)
	np.testing.assert_almost_equal(updated_particles[idx].acc, particles_test_data[idx].acc + 1, required_decimal_accuray)



# Test 3 - vector_difference
# pos vector - vel vector => store result in acc vector
particles_test_data = generateParticles(1)
expected_result = particles_test_data[0].pos - particles_test_data[0].vel

gpu_particles = ParticleGPUInterface(particles_test_data)
gpu_particles.cudaTests("vector_difference_test", len(particles_test_data))
updated_particles = gpu_particles.getResultsFromDevice()

# compare to the original python object
np.testing.assert_almost_equal(updated_particles[0].acc, expected_result, required_decimal_accuray)

# and compare to the particle passed back
np.testing.assert_almost_equal(updated_particles[0].acc, updated_particles[0].pos - updated_particles[0].vel, required_decimal_accuray)




# Sim specific code testing (methods in cuda_sim.c)
# Compare the results of the python implementations to the cuda ones

# Test 4 - pressure
from framework import pressure

particles_test_data = generateParticles(10)
pressures = [pressure(particle) for particle in particles_test_data]

gpu_particles = ParticleGPUInterface(particles_test_data)
gpu_particles.cudaTests("particle_pressure_test", len(particles_test_data))
updated_particles = gpu_particles.getResultsFromDevice()

for idx in xrange(len(particles_test_data)):
	np.testing.assert_almost_equal(updated_particles[idx].pressure, pressures[idx], required_decimal_accuray)

# Test 5 - Gaussian_kernel
from framework import Gaussian_kernel

particles_test_data = generateParticles(10)
gaussian_kernels = [Gaussian_kernel(particle.acc, particle.rho) for particle in particles_test_data]

gpu_particles = ParticleGPUInterface(particles_test_data)
gpu_particles.cudaTests("gaussian_kernel_test", len(particles_test_data))
updated_particles = gpu_particles.getResultsFromDevice()

for idx in xrange(len(particles_test_data)):
	np.testing.assert_almost_equal(updated_particles[idx].pressure, gaussian_kernels[idx], required_decimal_accuray)



print "All tests passed."