import numpy as np


# Updating particles
def predict_states(particle_states, contact_rate, recovery_rate):
    updated_particles = np.empty_like(particle_states)
    suscept, infect, recovered = particle_states[:, 0], particle_states[:, 1], particle_states[:, 2]
    total_pop = particle_states.sum(axis=-1)
    force_infection = contact_rate * infect / total_pop
    new_infections = np.random.binomial(suscept, 1.0 - np.exp(-force_infection))
    new_recoveries = np.random.binomial(infect, 1.0 - np.exp(-recovery_rate))

    updated_particles[:, 0] = suscept - new_infections
    updated_particles[:, 1] = infect + new_infections - new_recoveries
    updated_particles[:, 2] = recovered + new_recoveries

    return updated_particles
