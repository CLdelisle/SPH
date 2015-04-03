from particle import Particle
from gpu_interface import ParticleGPUInterface

particles = []

# Generate test particles
particles.append(Particle(2,2,2,2,2,2,2,2))
particles.append(Particle(3,3,3,3,3,3,3,3))
particles.append(Particle(4,4,4,4,4,4,4,4))
particles.append(Particle(5,5,5,5,5,5,5,5))
particles.append(Particle(6,6,6,6,6,6,6,6))

# Print initial particle states
for particle in particles:
	print particle.formatProperties()

# Initialize the gpu interface, pass in particles
gpu_particles = ParticleGPUInterface(particles)

# Calls the demo function on GPU, no data transfered back
# Formula for demo function: id[i] = mass[i] + pos_x[i]
gpu_particles.demo_particle_function()

# Transfer the results back to CPU
updated_particles = gpu_particles.getResultsFromDevice()

# Print final particle states
for particle in updated_particles:
	print particle.formatProperties()