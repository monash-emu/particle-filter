import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_counts_from_particles(particles):
    melted_df = pd.DataFrame(particles.T).melt(var_name="Columns", value_name="Values")
    return melted_df.groupby(["Columns", "Values"]).size().reset_index(name="Counts")


def get_links_from_pedigree(particles, pedigree, observations):
    links = []
    for t in range(len(observations)):
        for dest, origin in enumerate(pedigree[t, :]):
            links.append([t, t + 1, particles[t, 1, origin], particles[t + 1, 1, dest]])
    return np.array(links)


# Plot results for number of infectious from output array
def plot_particle_results(prop_particles, resamp_particles, observations):
    prop_counts = get_counts_from_particles(prop_particles[:, 1, :])
    resamp_counts = get_counts_from_particles(resamp_particles[:, 1, :])
    results_plot = plt.scatter(
        prop_counts["Columns"] - 0.2,
        prop_counts["Values"],
        s=prop_counts["Counts"] * 15.0,
        alpha=0.5,
        label="proposed",
    )
    plt.scatter(
        resamp_counts["Columns"],
        resamp_counts["Values"],
        s=resamp_counts["Counts"] * 15.0,
        alpha=0.5,
        label="resampled",
        color="g",
    )
    plt.scatter(range(1, len(observations) + 1), observations, label="target", color="k")
    plt.legend()
    return results_plot


def plot_links(particles, links, obs):
    resamp_counts = get_counts_from_particles(particles[:, 1, :])
    links_plot = plt.scatter(
        resamp_counts["Columns"],
        resamp_counts["Values"],
        s=resamp_counts["Counts"] * 15.0,
        alpha=0.5,
        label="resampled",
        color="g",
    )
    for l in links:
        plt.plot(l[:2], l[2:], color="gray", linewidth=0.1, label=None)
    plt.scatter(range(1, len(obs) + 1), obs, label="target", color="k")
    plt.legend()
    return links_plot
