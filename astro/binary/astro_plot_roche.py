#!/usr/bin/env python 

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
from scipy.optimize import fsolve
from matplotlib.colors import LogNorm
import math
# 定数
G = 6.67430e-11  # 重力定数 [m^3/kg/s^2]
AU = 1.496e11  # 天文単位 [m]
SOLAR_RADIUS = 6.957e8  # 太陽半径 [m]
DAY_TO_SECOND = 86400  # 1日を秒に変換

def grad_potential_x(x, y, mu):
    """
    Calculate the x-component of the gradient of the Roche potential.

    Parameters
    ----------
    x : float
        x-coordinate [m].
    y : float
        y-coordinate [m].
    mu : float
        Reduced mass parameter (mass ratio of the secondary body).

    Returns
    -------
    gradient : float
        The x-component of the gradient of the Roche potential.
    """
    r1 = np.sqrt((x + mu)**2 + y**2)
    r2 = np.sqrt((x - (1 - mu))**2 + y**2)
    return (1 - mu) * (x + mu) / r1**3 + mu * (x - (1 - mu)) / r2**3 - x

def find_lagrange_points(mu):
    """
    Calculate the positions of the Lagrange points (L1, L2, L3).

    Parameters
    ----------
    mu : float
        Reduced mass parameter (mass ratio of the secondary body).

    Returns
    -------
    lagrange_points : list of float
        Positions of L1, L2, and L3 along the x-axis [m].
    """
    print("Finding Lagrange points...")
    guesses = [0.5 - mu, 1.5 - mu, -1.5 - mu]
    lagrange_points = []
    for guess in guesses:
        sol = fsolve(lambda x: grad_potential_x(x, 0, mu), guess)
        lagrange_points.append(sol[0])
        print(f"  Initial guess: {guess:.2f}, Lagrange point: {sol[0]:.5e} [m]")
    return lagrange_points

def calculate_mean_distance(m1, m2, period_days):
    """
    Calculate the mean distance between two bodies in a binary system.

    Parameters
    ----------
    m1 : float
        Mass of the primary body [kg].
    m2 : float
        Mass of the secondary body [kg].
    period_days : float
        Orbital period of the system [days].

    Returns
    -------
    distance : float
        Mean distance between the two bodies [m].
    """
    print(f"Calculating mean distance for orbital period: {period_days} days")
    period_seconds = period_days * DAY_TO_SECOND
    total_mass = m1 + m2
    distance = ((G * total_mass * period_seconds**2) / (4 * np.pi**2))**(1/3)
    print(f"  Mean distance: {distance:.5e} [m]")
    return distance

def calculate_roche_potential(X, Y, m1, m2, distance):
    """
    Calculate the Roche potential for a binary system at given points (X, Y).

    Parameters
    ----------
    X : ndarray
        x-coordinates of the grid points [m].
    Y : ndarray
        y-coordinates of the grid points [m].
    m1 : float
        Mass of the primary body [kg].
    m2 : float
        Mass of the secondary body [kg].
    distance : float
        Distance between the centers of the two bodies [m].

    Returns
    -------
    potential : ndarray
        The Roche potential at each grid point [J/kg].
    """
    print("Calculating Roche potential...")
    mu = m2 / (m1 + m2)
    omega = np.sqrt(G * (m1 + m2) / distance**3)
    print(f"  Reduced mass parameter (mu): {mu:.5f}")
    print(f"  Orbital angular velocity (omega): {omega:.5e} [rad/s]")

    r1 = np.sqrt((X + mu * distance)**2 + Y**2)
    r2 = np.sqrt((X - (1 - mu) * distance)**2 + Y**2)
    potential = -G * m1 / r1 - G * m2 / r2 - 0.5 * omega**2 * (X**2 + Y**2)
    print("  Roche potential calculation completed.")
    return potential


def plot_physical_potential(
    m1, m2, distance, unit="m", star_radius=None, plot_center=False, title=None, 
    savefig=None, plot_detail=True, contour_levels=50, contour_levels_line=50, star_radius_tangent=False
):
    """
    Plot the Roche potential and the positions of the Lagrange points.
    Optionally display additional details, centers, and save the plot.

    Parameters
    ----------
    m1 : float
        Mass of the primary body [kg].
    m2 : float
        Mass of the secondary body [kg].
    distance : float
        Distance between the two bodies [m].
    unit : str, optional
        Unit for the plot axes ('m', 'AU', or 'solar_radius'). Default is 'm'.
    star_radius : float, optional
        Radius of the secondary star [m]. If specified, a circle representing the star is plotted.
    plot_center : bool, optional
        If True, marks the positions of the centers (m1, m2) and the barycenter.
    title : str, optional
        Title of the plot. If None, no title is set.
    savefig : str, optional
        Path to save the figure. If None, the plot is not saved.
    plot_detail : bool, optional
        If True, displays additional details (masses, distance, etc.) on the plot.
    contour_levels : int, optional
        Number of levels for the contour plot. Default is 50.
    contour_levels_line : int, optional
        Number of levels for the contour plot for a line style. Default is 50.
    star_radius_tangent : bool, optional
        If True, plot tangential lines at the star surface 

    Raises
    ------
    ValueError
        If an invalid unit is provided.
    """
    print("Starting Roche potential plot...")
    print("  Coordinate system origin: barycenter of the system.")

    # Unit conversion
    units_conversion = {"m": 1, "AU": AU, "solar_radius": SOLAR_RADIUS}
    if unit not in units_conversion:
        raise ValueError(f"Invalid unit: {unit}. Choose from {list(units_conversion.keys())}.")

    scale = units_conversion[unit]
    mu = m2 / (m1 + m2)

    # Create grid
    x = np.linspace(-1.5 * distance, 1.5 * distance, 500) / scale
    y = np.linspace(-1.5 * distance, 1.5 * distance, 500) / scale
    X, Y = np.meshgrid(x, y)

    # Calculate Roche potential
    Z = calculate_roche_potential(X * scale, Y * scale, m1, m2, distance)
    Z_posi = -Z

    print("  Plotting the Roche potential...")
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    levels = np.logspace(np.log10(Z_posi.min()), np.log10(Z_posi.max()), contour_levels)
    contour = ax.contourf(X, Y, Z_posi, levels=levels, cmap='cool', norm=LogNorm(), alpha=0.2)
    cbar = plt.colorbar(contour, ax=ax, format="%.1e", shrink=0.62)
    cbar.set_label("Roche Potential (J/kg)")
    # add coutour 
    levels_line = np.logspace(np.log10(Z_posi.min()), np.log10(Z_posi.max()), contour_levels_line)    
    ax.contour(X, Y, Z_posi, levels=levels_line, colors="grey", linewidths=0.5, alpha=0.2)


    # Plot Lagrange points
    L1, L2, L3 = [point * distance / scale for point in find_lagrange_points(mu)]
    ax.scatter([L1], [0], color="black", label="L1", marker=".", alpha=0.6)
    ax.scatter([L2, L3], [0, 0], color="grey", label="L2, L3", marker=".", alpha=0.8)

    L4_x, L4_y = 0.5 * distance / scale - mu * distance / scale, np.sqrt(3) / 2 * distance / scale
    L5_x, L5_y = 0.5 * distance / scale - mu * distance / scale, -np.sqrt(3) / 2 * distance / scale
    ax.scatter([L4_x, L5_x], [L4_y, L5_y], color="grey", label="L4, L5", marker=".", alpha=0.3)

    # Optionally plot the secondary star's radius
    if star_radius is not None:
        star_radius_scaled = star_radius / scale
        r2_scaled = (1 - mu) * distance / scale  # Position of the secondary star
        print(f"  Plotting secondary star with radius: {star_radius_scaled:.5f} {unit}")
        circle = plt.Circle(
            (r2_scaled, 0),  # Center of the secondary star
            star_radius_scaled,  # Radius in scaled units
            color="blue",
            alpha=0.3,
            label=fr"Star Radius ({unit})",
        )
        ax.add_artist(circle)

    # Optionally plot the secondary star's tangent line 
    if star_radius_tangent:
        # Calculate tangent angle
        d_scaled = distance / scale  # Scaled distance between stars
        if star_radius_scaled < d_scaled:
            theta = np.arcsin(star_radius_scaled / d_scaled)  # Angle in radians
            print(f"  Tangent angle (deg): {np.degrees(theta):.2f}")

            # Tangent line from primary star
            r1_scaled = -mu * distance / scale  # Position of the primary star
            tangent_x = [r1_scaled, r2_scaled - star_radius_scaled * np. sin(theta)]
            tangent_y = [0, star_radius_scaled * np.cos(theta)]
            ax.plot(tangent_x, tangent_y, color="cyan", linestyle="--", label=f"Inclination (deg): {90 - np.degrees(theta):.2f}")

            tangent_x_45 = [r1_scaled, r2_scaled]
            tangent_y_45 = [        0, d_scaled]
            ax.plot(tangent_x_45, tangent_y_45, color="green", linestyle="--", label=f"Inclination (deg): 45")

            tangent_x_30 = [r1_scaled, r1_scaled + 0.8 * d_scaled]
            tangent_y_30 = [        0, 0.8 * math.sqrt(3) * d_scaled]
            ax.plot(tangent_x_30, tangent_y_30, color="yellow", linestyle="--", label=f"Inclination (deg): 30")

            if title=="Centaurus X-3":
                tangent_x_70 = [r1_scaled, r2_scaled]
                # inclintion 70度は、星からみると20度なので、20度を変換
                angle_deg = 90 - 70 
                angle_rad = math.radians(angle_deg)  # 度をラジアンに変換
                # tan(70°) を計算
                tan_70 = math.tan(angle_rad)                
                tangent_y_70 = [        0, d_scaled * tan_70]
                ax.plot(tangent_x_70, tangent_y_70, color="magenta", linestyle="--", label=f"Inclination (deg): 70")


    # Optionally add plot title
    if title:
        ax.set_title(title)

    # Optionally display plot details
    if plot_detail:
        detail_text = (
            f"Mass 1 (Primary): {m1:.2e} kg\n"
            f"Mass 2 (Secondary): {m2:.2e} kg\n"
            f"Distance: {distance:.2e} m\n"
            f"Period: {np.sqrt(4 * np.pi**2 * distance**3 / (G * (m1 + m2))) / DAY_TO_SECOND:.2f} days\n"
            f"Lagrange Points ({unit}):\n  L1: {L1:.2f}, L2: {L2:.2f}, L3: {L3:.2f}"
        )
        fig.text(0.02, 0.9, detail_text, fontsize=9, va="center", ha="left", bbox=dict(boxstyle="round", alpha=0.3))

    # Axes labels and grid
    ax.set_xlabel(f"x ({unit})")
    ax.set_ylabel(f"y ({unit})")
    plt.grid(True, alpha=0.01)

    # Optionally plot centers and barycenter
    if plot_center:
        r1_scaled = -mu * distance / scale  # Position of the primary star
        r2_scaled = (1 - mu) * distance / scale  # Position of the secondary star
        ax.scatter([0], [0], color="red", label="Barycenter", marker="x", s=50)
        ax.scatter([r1_scaled], [0], color="orange", label="Primary Center", marker="x", s=50)
        ax.scatter([r2_scaled], [0], color="blue", label="Secondary Center", marker="x", s=50)

#    ax.legend()
    # Configure the legend
    ax.legend(
        loc='upper right',  # Place the legend in the upper right corner
        bbox_to_anchor=(1.1, 1.3),  # Slightly outside the plot
        fontsize='small',  # Set a smaller font size
        frameon=True,  # Optional: add a border around the legend
        fancybox=True,  # Optional: rounded border
        shadow=True  # Optional: shadow effect for better visibility
    )

    # Save the plot if savefig is specified
    if savefig:
        plt.savefig(savefig)
        print(f"Plot saved to {savefig}")

    plt.show()
    print("Plot completed.")

if __name__ == "__main__":

    # Example: Earth-Sun system
    M_sun = 1.989e30  # Mass of the Sun [kg]
    M_earth = 5.972e24  # Mass of the Earth [kg]
    distance_earth_sun = AU  # Mean distance between Earth and Sun [m]
    plot_physical_potential(M_sun, M_earth, distance_earth_sun, unit="AU", title="Sun - Earth", savefig="roche_earth.png", \
       contour_levels=100, contour_levels_line=400)

    # Example: Earth-Sun system
    M_sun = 1.989e30  # Mass of the Sun [kg]
    M_jupiter = 1.898e27 # Mass of the Jupiter [kg]
    distance_jupiter_sun = 5.20260 * AU  # Mean distance between Jupiter and Sun [m]
    plot_physical_potential(M_sun, M_jupiter, distance_jupiter_sun, unit="AU", \
    star_radius=None, plot_center=True, title="Jupiter", savefig="roche_jupiter.png", \
    contour_levels=100, contour_levels_line=400)

    # Example: Cyg X-1 system
    period_days = 5.6  # Orbital period [days]
    M_bh = 21.2 * M_sun  # Black hole mass [kg]
    M_star = 40.6 * M_sun  # Companion star mass [kg]
    R_star = 22.3 * SOLAR_RADIUS # Companion star radius [m]
    distance_cygx1 = calculate_mean_distance(M_bh, M_star, period_days)
    plot_physical_potential(M_bh, M_star, distance_cygx1, unit="solar_radius", star_radius=R_star, \
    plot_center=True, title="Cygnus X-1", savefig="roche_cygx1.png",\
    contour_levels=100, contour_levels_line=400, star_radius_tangent=True)


    # Example: Centaurus X-3 
    period_days = 2.08  # Orbital period [days]
    M_ns = 1.2 * M_sun  # Neutron Star mass [kg]
    M_star = 40.6 * M_sun  # Companion star mass [kg]
    R_star = 11.8 * SOLAR_RADIUS # Companion star radius [m]
    distance_cenx3 = calculate_mean_distance(M_ns, M_star, period_days)
    plot_physical_potential(M_ns, M_star, distance_cenx3, unit="solar_radius", star_radius=R_star, \
    plot_center=True, title="Centaurus X-3", savefig="roche_cenx3.png",\
    contour_levels=100, contour_levels_line=400, star_radius_tangent=True)


    # Example: 4U1700-377
    period_days = 3.41  # Orbital period [days]
    M_ns = 2.44 * M_sun  # Neutron Star mass [kg]
    M_star = 34 * M_sun  # Companion star mass [kg]
    R_star = 19 * SOLAR_RADIUS # Companion star radius [m]
    distance_cenx3 = calculate_mean_distance(M_ns, M_star, period_days)
    plot_physical_potential(M_ns, M_star, distance_cenx3, unit="solar_radius", star_radius=R_star, \
    plot_center=True, title="4U1700-377", savefig="roche_4u1700-377.png",\
    contour_levels=100, contour_levels_line=400, star_radius_tangent=True)
