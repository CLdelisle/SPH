from particle import Particle
from gpu_interface import ParticleGPUInterface

# Generate test particles
particles = [
	Particle(2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8),
	Particle(3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8),
	Particle(4.1,4.2,4.3,4.4,4.5,4.6,4.7,4.8),
	Particle(5.1,5.2,5.3,5.4,5.5,5.6,5.7,5.8),
	Particle(6.1,6.2,6.3,6.4,6.5,6.6,6.7,6.8)
]
# Initialize the gpu interface, pass in particles
gpu_particles = ParticleGPUInterface(particles)

gpu_particles.cudaTests(len(particles))

# Transfer the results back to CPU
updated_particles = gpu_particles.getResultsFromDevice()