from particle import Particle
from gpu_interface import ParticleGPUInterface
import numpy
import random
import numpy as np

def randomFloat():
	return random.uniform(1, 1000)

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


np.testing.assert_almost_equal(updated_particle.id, particles_test_data[0].id + 1)
np.testing.assert_almost_equal(updated_particle.mass, numpy.float32(particles_test_data[0].mass) + 1)
np.testing.assert_almost_equal(updated_particle.pos, np.array(particles_test_data[0].pos, dtype=np.float32) + 1)
np.testing.assert_almost_equal(updated_particle.vel, np.array(particles_test_data[0].vel, dtype=np.float32) + 1)



# Test 2 - Same as test 1, but apply +1 to all properties on multiple particles. Tests thread access index
particles_test_data = generateParticles(5)

gpu_particles = ParticleGPUInterface(particles_test_data)
gpu_particles.cudaTests("increment_particle_properties_on_multiple_particles", len(particles_test_data))
updated_particles = gpu_particles.getResultsFromDevice()

for idx in xrange(len(particles_test_data)):
	np.testing.assert_almost_equal(updated_particles[idx].id, particles_test_data[idx].id + 1, 3)
	np.testing.assert_almost_equal(updated_particles[idx].mass, numpy.float32(particles_test_data[idx].mass) + 1, 3)
	np.testing.assert_almost_equal(updated_particles[idx].pos, np.array(particles_test_data[idx].pos, dtype=np.float32) + 1, 3)
	np.testing.assert_almost_equal(updated_particles[idx].vel, np.array(particles_test_data[idx].vel, dtype=np.float32) + 1, 3)



# Test 3 - vector_difference
# pos vector - vel vector => store result in acc vector

particles_test_data = generateParticles(1)
expected_result = particles_test_data[0].pos - particles_test_data[0].vel

gpu_particles = ParticleGPUInterface(particles_test_data)
gpu_particles.cudaTests("vector_difference_test", len(particles_test_data))
updated_particles = gpu_particles.getResultsFromDevice()

# compare to the original python object
np.testing.assert_almost_equal(updated_particles[0].acc, np.array(expected_result, dtype=np.float32), 3)

# and compare to the particle passed back
np.testing.assert_almost_equal(updated_particles[0].acc, np.array(updated_particles[0].pos - updated_particles[0].vel, dtype=np.float32), 3)

print "All tests passed."