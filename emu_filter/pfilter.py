import numpy as np


def filter_loop(observations, predict_state, particles, importance, params, target_sd):
    prop_particles = np.copy(particles)
    n_particles = particles.shape[2]
    pedigree = np.zeros([len(observations), n_particles], dtype=int)
    for t, obs in enumerate(observations):
    
        # Prediction
        proposed_particles = predict_state(particles[t, :, :], **params)
        prop_particles[t + 1, :] = proposed_particles
    
        # Importance
        weights = importance(proposed_particles[1, :], obs, target_sd)
        norm_weights = weights / sum(weights)
    
        # Resampling
        indices = np.random.choice(n_particles, size=n_particles, p=norm_weights)
        pedigree[t, :] = indices
        resamp_particles = proposed_particles[:, indices]
        
        # Update
        particles[t + 1, :, :] = resamp_particles

    return particles, prop_particles, pedigree