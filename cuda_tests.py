from particle import Particle
from gpu_interface import ParticleGPUInterface
import numpy
import numpy as np

# Test 1 - increment all particle property values by 1
particles_test_data = [Particle(89, 2.223, 52.3, 12.4, 72.5, 2.556, 2.25637, 2.80123)]
gpu_particles = ParticleGPUInterface(particles_test_data)
gpu_particles.cudaTests("increment_particle_properties", len(particles_test_data))
updated_particle = gpu_particles.getResultsFromDevice()[0]


np.testing.assert_almost_equal(updated_particle.id, particles_test_data[0].id + 1)
np.testing.assert_almost_equal(updated_particle.mass, numpy.float32(particles_test_data[0].mass) + 1)
np.testing.assert_almost_equal(updated_particle.pos, np.array(particles_test_data[0].pos, dtype=np.float32) + 1)
np.testing.assert_almost_equal(updated_particle.vel, np.array(particles_test_data[0].vel, dtype=np.float32) + 1)



# Test 2 - Same as test 1, but apply +1 to all properties on multiple particles. Tests thread access index
particles_test_data = [
	Particle(2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8),
	Particle(3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8),
	Particle(4.1,4.2,4.3,4.4,4.5,4.6,4.7,4.8),
	Particle(5.1,5.2,5.3,5.4,5.5,5.6,5.7,5.8),
	Particle(6.1,6.2,6.3,6.4,6.5,6.6,6.7,6.8)
]

gpu_particles = ParticleGPUInterface(particles_test_data)
gpu_particles.cudaTests("increment_particle_properties_on_multiple_particles", len(particles_test_data))
updated_particles = gpu_particles.getResultsFromDevice()

for idx in xrange(len(particles_test_data)):
	np.testing.assert_almost_equal(updated_particles[idx].id, particles_test_data[idx].id + 1, 5)
	np.testing.assert_almost_equal(updated_particles[idx].mass, numpy.float32(particles_test_data[idx].mass) + 1, 5)
	np.testing.assert_almost_equal(updated_particles[idx].pos, np.array(particles_test_data[idx].pos, dtype=np.float32) + 1, 5)
	np.testing.assert_almost_equal(updated_particles[idx].vel, np.array(particles_test_data[idx].vel, dtype=np.float32) + 1, 5)



print "All tests passed."