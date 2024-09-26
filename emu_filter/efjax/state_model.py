import numpy as np
from jax import numpy as jnp
from jax import random


# Updating particles
def ensemble_step(particles, prng_key, *args):

    ki, kr = random.split(prng_key, 2)

    contact_rate, recovery_rate, total_pop = args
    updated_particles = jnp.empty_like(particles)
    suscept, infect, recovered = particles[0], particles[1], particles[2]
    new_infections = random.binomial(ki, suscept, 1.0 - jnp.exp(-contact_rate * infect / total_pop)).astype(jnp.int32)
    new_recoveries = random.binomial(kr, infect, 1.0 - jnp.exp(-recovery_rate)).astype(jnp.int32)

    updated_particles = updated_particles.at[0, :].set(suscept - new_infections)
    updated_particles = updated_particles.at[1, :].set(infect + new_infections - new_recoveries)
    updated_particles = updated_particles.at[2, :].set(recovered + new_recoveries)

    return updated_particles
