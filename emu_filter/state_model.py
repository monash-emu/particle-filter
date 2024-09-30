import numpy as np


# Updating particles
# def predict_states(particles, contact_rate, recovery_rate, total_pop):
#     updated_particles = np.empty_like(particles)
#     suscept, infect, recovered = particles[0, :], particles[1, :], particles[2, :]
#     new_infections = np.random.binomial(suscept, 1.0 - np.exp(-contact_rate * infect / total_pop))
#     new_recoveries = np.random.binomial(infect, 1.0 - np.exp(-recovery_rate))

#     updated_particles[0, :] = suscept - new_infections
#     updated_particles[1, :] = infect + new_infections - new_recoveries
#     updated_particles[2, :] = recovered + new_recoveries

#     return updated_particles


def predict_states(particles, contact_rate, recovery_rate, total_pop, volatility):
    updated_particles = np.empty_like(particles)
    suscept, infect, recovered, process = particles[0, :], particles[1, :], particles[2, :], particles[3, :]
    new_contact_rates = np.exp(np.random.normal(np.log(process), volatility, particles.shape[1]))
    new_infections = new_contact_rates * suscept * infect / total_pop
    new_recoveries = recovery_rate * infect

    updated_particles[0, :] = suscept - new_infections
    updated_particles[1, :] = infect + new_infections - new_recoveries
    updated_particles[2, :] = recovered + new_recoveries
    updated_particles[3, :] = new_contact_rates
    
    return updated_particles
