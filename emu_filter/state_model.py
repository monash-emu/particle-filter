import numpy as np


# Updating particles
def predict_states(
    particle_states: np.array, 
    contact_rate: float, 
    recovery_rate: float,
) -> np.array:
    """Update particle states using a standard SIR model
    under a frequency-dependent transmission assumption.

    Args:
        particle_states: The previous states
        contact_rate: The per capita effective contact rate
        recovery_rate: The recovery rate

    Returns:
        The updated states for the particles
    """
    suscept, infect = particle_states[:, 0], particle_states[:, 1]
    total_pop = particle_states.sum(axis=-1)
    force_infection = contact_rate * infect / total_pop
    new_infections = np.random.binomial(suscept, 1.0 - np.exp(-force_infection))
    new_recoveries = np.random.binomial(infect, 1.0 - np.exp(-recovery_rate))

    updated_particles = particle_states.copy()
    updated_particles[:, 0] -= new_infections
    updated_particles[:, 1] += new_infections - new_recoveries
    updated_particles[:, 2] += new_recoveries
    return updated_particles
