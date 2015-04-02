from particle import Particle
from gpu_interface import ParticleGPUInterface

particles = []
particles.append(Particle(2,2,2,2,2,2,2,2))
particles.append(Particle(3,3,3,3,3,3,3,3))
particles.append(Particle(4,4,4,4,4,4,4,4))
particles.append(Particle(5,5,5,5,5,5,5,5))
particles.append(Particle(6,6,6,6,6,6,6,6))

tester = ParticleGPUInterface(particles)

tester.double_array()