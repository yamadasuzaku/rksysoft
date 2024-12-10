#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

# Set font properties for plots
plt.rcParams['font.family'] = 'serif'

# Define constants
INCLINATION = 30  # Viewing inclination in degrees

# Stellar wind parameters for different models (stellar radius scaling factor)
parameters = {
    0.5: {
        0: {"r_star": 1.496e12, "a": 1.60, "v_inf": 2540, "rho_0": 6.17e-14},
        20: {"r_star": 1.387e12, "a": 1.05, "v_inf": 1580, "rho_0": 3.72e-14},
    },
    0.4: {
        0: {"r_star": 1.197e12, "a": 1.16, "v_inf": 2030, "rho_0": 4.01e-14},
        20: {"r_star": 1.157e12, "a": 0.96, "v_inf": 1650, "rho_0": 4.61e-14},
    },
}

def interpolate(value_0, value_20, theta):
    """Interpolates parameter values for a given angle θ between 0° and 20°."""
    return value_0 + (value_20 - value_0) * (theta / 20) ** 2

def v_wind(r, theta, stellar_radius):
    """Calculates the wind velocity at a given radius and angle."""
    params = parameters[stellar_radius]
    r_star_theta = interpolate(params[0]["r_star"], params[20]["r_star"], theta)
    a_theta = interpolate(params[0]["a"], params[20]["a"], theta)
    v_inf_theta = interpolate(params[0]["v_inf"], params[20]["v_inf"], theta)    

    if 1 - r_star_theta / r > 0:
        velecoty_wind = v_inf_theta * (1 - r_star_theta / r) ** a_theta
    else:
        velecoty_wind = 0
    return velecoty_wind

def rho_wind(r, theta, stellar_radius):
    """Calculates the wind density at a given radius and angle."""
    params = parameters[stellar_radius]
    r_star_theta = interpolate(params[0]["r_star"], params[20]["r_star"], theta)
    rho_0_theta = interpolate(params[0]["rho_0"], params[20]["rho_0"], theta)
    a_theta = interpolate(params[0]["a"], params[20]["a"], theta)
    numerator = (r_star_theta / r) ** 2 * rho_0_theta
#    print(f"r_star_theta / r = {r_star_theta} / {r}, a_theta = {a_theta}")
    if 1 - (r_star_theta / r) > 0:
        denominator = (1 - (r_star_theta / r)) ** a_theta
        adensity_wind = numerator / denominator
    else:
        adensity_wind = 0
    return adensity_wind

def plot_v_rho(stellar_radius):
    """Generates and saves plots for wind velocity, density profiles, and projected velocities."""
    solar_radius = 6.96e10  # Solar radius in cm
    stellar_radius_in_solar = 22  # Stellar radius in units of solar radius
    r_values = np.linspace(solar_radius, 50 * solar_radius, 500)
    theta_values = [0, 5, 10, 15, 20]

    # Plot wind velocity and density profiles
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Stellar Radius = {stellar_radius} binary separation (d)"  + f"{stellar_radius_in_solar}  " + r"$R_{star}$=" + f"{stellar_radius_in_solar}" + r"$R_{\odot}$", fontsize=12)

    # Wind velocity profiles
    ax1 = axes[0]
    ax2 = ax1.secondary_xaxis('top', functions=(lambda x: x / stellar_radius_in_solar, 
                                                lambda x: x * stellar_radius_in_solar))
    ax2.set_xlabel(r"$R$ [$R_{star}$]")
    for theta in theta_values:
        v_values = [v_wind(r, theta, stellar_radius) for r in r_values]
        line, = ax1.plot(r_values / solar_radius, v_values, label=f"Theta = {theta} deg")

        v_inf_theta = interpolate(
            parameters[stellar_radius][0]["v_inf"],
            parameters[stellar_radius][20]["v_inf"],
            theta,
        )
        ax1.axhline(y=v_inf_theta, color=line.get_color(), linestyle="--", label=f"v_inf ({theta} deg)")

    ax1.set_xlabel(r"$R$ [$R_{\odot}$]")
    ax1.set_ylabel(r"$v_{wind}$ [km/s]")
    ax1.legend()
    ax1.grid(alpha=0.5)

    # Wind density profiles
    ax3 = axes[1]
    ax4 = ax3.secondary_xaxis('top', functions=(lambda x: x / stellar_radius_in_solar, 
                                                lambda x: x * stellar_radius_in_solar))
    ax4.set_xlabel(r"$R$ [$R_{star}$]")

    m_H = 1.67e-24  # Hydrogen atom mass [g]
    mu = 1  # Mean molecular weight
    ax5 = ax3.secondary_yaxis('right', functions=(lambda rho: rho / (mu * m_H),
                                                  lambda n: n * (mu * m_H)))
    ax5.set_ylabel(r"$n_{wind}$ [cm$^{-3}$]")

    for theta in theta_values:
        rho_values = [rho_wind(r, theta, stellar_radius) for r in r_values]
        ax3.plot(r_values / solar_radius, rho_values, label=f"Theta = {theta} deg")
    ax3.set_xlabel(r"$R$ [$R_{\odot}$]")
    ax3.set_ylabel(r"$\rho_{wind}$ [g/cm³]")
    ax3.set_yscale("log")
    ax3.legend()
    ax3.grid()

    plt.tight_layout()
    plt.savefig(f"stellar_profile_{stellar_radius}.png", bbox_inches='tight')
    plt.show()

    # Plot projected wind velocities
    r_values_forproj = np.linspace(solar_radius * 25, 50 * solar_radius, 5)
    fig, axes = plt.subplots(len(theta_values), 1, figsize=(10, 8), sharex=True)
    fig.subplots_adjust(right=0.8)
    fig.suptitle(f"Projected Velocities: Stellar Radius = {stellar_radius} binary separation (d) \n" + f"inclination = {INCLINATION} (deg)   " + r"$R_{star}$=" + f"{stellar_radius_in_solar}" + r"$R_{\odot}$", fontsize=12)

    for i, theta in enumerate(theta_values):
        v_values = [v_wind(r, theta, stellar_radius) for r in r_values_forproj]
        for v, r in zip(v_values, r_values_forproj):
            phase = np.linspace(0, 4 * np.pi, 100)
            view_angle = -np.cos(phase) * np.deg2rad(INCLINATION)
            v_proj = v * np.sin(np.deg2rad(theta) - view_angle)
            axes[i].plot(phase / (2 * np.pi), v_proj, label=f"$\\theta$={theta} deg, $R={r / solar_radius:.1f}$ Rsun")
            axes[i].set_ylabel(r"$v_{proj}$ [km/s]")
            axes[i].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
            axes[i].grid(alpha=0.3)
            axes[i].set_ylim(-1000,1000)

    axes[-1].set_xlabel("Orbital Phase")
    plt.tight_layout()
    plt.savefig(f"stellar_profile_{stellar_radius}_proj.png", bbox_inches='tight')
    plt.show()

# Main execution
if __name__ == "__main__":
    for stellar_radius in [0.5, 0.4]: # stallar radius in unit of binary separation 
        plot_v_rho(stellar_radius)
