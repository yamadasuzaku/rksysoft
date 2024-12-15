import numpy as np
import matplotlib.pyplot as plt

# 定数の定義
G = 6.67430e-11  # Gravitational constant [m^3 kg^-1 s^-2]
DAY_TO_SECOND = 86400  # Days to seconds conversion
M_sun = 1.989e30  # Solar mass [kg]
SOLAR_RADIUS = 6.957e8  # Solar radius [m]

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
    period_seconds = period_days * DAY_TO_SECOND
    total_mass = m1 + m2
    distance = ((G * total_mass * period_seconds**2) / (4 * np.pi**2))**(1/3)
    return distance

def plot_binary_system_with_velocity(
    M_bh, M_star, R_star, distance_cygx1, grid_size=30, grid_range_factor=2, \
    v_scale=0.05, cmap='coolwarm_r', plot_star_radius=True, mode="star_wind", axis_symmetric=False):
    """
    Plot the velocity field of a binary system with the center of mass at the origin.

    Parameters
    ----------
    M_bh : float
        Black hole mass [kg].
    M_star : float
        Companion star mass [kg].
    R_star : float
        Companion star radius [m].
    distance_cygx1 : float
        Mean distance between the two bodies [m].
    grid_size : int, optional
        Resolution of the grid. Default is 50.
    grid_range_factor : float, optional
        Factor for setting the grid range relative to the binary distance. Default is 2.
    v_scale : float, optional
        Scaling factor for the velocity field. Default is 0.05.
    cmap : str, optional
        Colormap for the velocity field. Default is 'coolwarm'.
    plot_star_radius : bool, optional
        Whether to plot the radius of the companion star. Default is True.
    mode : str, optional
        Velocity field mode. One of {"star_wind", "towards_bh", "perpendicular_bh"}.
    axis_symmetric : bool, optional
        If True, apply axis-symmetric scaling (cos2 theta) dependence). Default is False.
    """

    # 共通重心を原点とした座標系での位置
    bh_x = -distance_cygx1 * (M_star / (M_bh + M_star))
    star_x = distance_cygx1 * (M_bh / (M_bh + M_star))

    # 描画範囲
    grid_range = grid_range_factor * distance_cygx1
    x = np.linspace(-grid_range, grid_range, grid_size)
    y = np.linspace(-grid_range, grid_range, grid_size)
    X, Y = np.meshgrid(x, y)

    # 伴星からの距離 r を計算
    R_star_grid = np.sqrt((X - star_x)**2 + Y**2)
    print(f"R_star_grid={R_star_grid}")

    # 球対称の速度場（v ∝ r）
    Vx = (X - star_x) / R_star_grid  # x方向速度
    Vy = Y / R_star_grid  # y方向速度
    Vx[R_star_grid < R_star] = 0  # 伴星内部では速度を0に設定
    Vy[R_star_grid < R_star] = 0

    # 軸対称依存性の適用
    if axis_symmetric:
        # ブラックホール方向をゼロとした角度 \(\theta\) を計算
        n_x = bh_x - X
        n_y = -Y
        n_mag = np.sqrt(n_x**2 + n_y**2)
        cos_theta = (n_x * Vx + n_y * Vy) / (n_mag * np.sqrt(Vx**2 + Vy**2))
        cos_theta[np.isnan(cos_theta)] = 0  # 0除算を防ぐ
        scaling = np.clip(cos_theta**2, 0, 1)  # \(\cos^2\theta\) をスケーリング因子として適用
        Vx *= scaling
        Vy *= scaling

    # ブラックホール方向の単位ベクトル
    n_x = bh_x - X
    n_y = -Y
    n_mag = np.sqrt(n_x**2 + n_y**2)
    n_x /= n_mag
    n_y /= n_mag

    # ブラックホール方向の速度成分（符号付き）
    V_bh_magnitude = Vx * n_x + Vy * n_y
    V_bh_x = V_bh_magnitude * n_x
    V_bh_y = V_bh_magnitude * n_y

    # ブラックホール垂直方向の速度成分（大きさ）
    V_perp_x = Vx - V_bh_x
    V_perp_y = Vy - V_bh_y
    V_perp_magnitude = np.sqrt(V_perp_x**2 + V_perp_y**2)

    # ブラックホール垂直方向の速度成分（符号付き）
    V_perp_x = Vx - V_bh_x
    V_perp_y = Vy - V_bh_y
    V_perp_magnitude = Vx * n_y - Vy * n_x  # 外積に基づいた符号付きの計算

    # プロットする速度場を選択
    if mode == "star_wind":
        plot_Vx, plot_Vy, color_data, title = Vx, Vy, np.sqrt(Vx**2 + Vy**2), "Star Wind Velocity Field"
        filename = "cygx1_star_wind_velocity.png" if not axis_symmetric else "cygx1_star_wind_velocity_axis_symmetric.png"
    elif mode == "towards_bh":
        plot_Vx, plot_Vy, color_data, title = V_bh_x, V_bh_y, V_bh_magnitude, "Velocity Towards Black Hole"
        filename = "cygx1_towards_bh_velocity.png" if not axis_symmetric else "cygx1_towards_bh_velocity_axis_symmetric.png"
    elif mode == "perpendicular_bh":
        plot_Vx, plot_Vy, color_data, title = V_perp_x, V_perp_y, V_perp_magnitude, "Perpendicular Velocity Field"
        filename = "cygx1_perpendicular_bh_velocity.png" if not axis_symmetric else "cygx1_perpendicular_bh_velocity_axis_symmetric.png"
    else:
        raise ValueError(f"Invalid mode: {mode}. Choose from 'star_wind', 'towards_bh', 'perpendicular_bh'.")

    # 描画
    fig, ax = plt.subplots(figsize=(8, 8))
    quiver = ax.quiver(
        X, Y, plot_Vx, plot_Vy, color_data,
        cmap=cmap, scale_units='xy', angles='xy'
    )
    # quiver = ax.quiver(
    #     X, Y, plot_Vx, plot_Vy, color_data,
    #     cmap=cmap, scale=1, scale_units='xy', angles='xy'
    # )

    print("X = ", X)
    print("plot_Vx =", plot_Vx)

    ax.plot(bh_x, 0, 'ko', markersize=10, label='Black Hole')  # ブラックホール
    ax.plot(star_x, 0, 'ro', markersize=10, label='Companion Star')  # 伴星

    # 伴星の半径を描画
    if plot_star_radius:
        circle = plt.Circle(
            (star_x, 0),  # 中心座標
            R_star,  # 半径
            color='blue',
            alpha=0.3,
            label="Companion Star Radius"
        )
        ax.add_artist(circle)

    # 描画設定
    ax.set_xlim(-grid_range, grid_range)
    ax.set_ylim(-grid_range, grid_range)
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_title(title)
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.legend()
    ax.grid()

    # Colorbar
    cbar = plt.colorbar(quiver, ax=ax)
    cbar.set_label('Velocity Magnitude [m/s]')

    # 保存
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Plot saved as {filename}")
    plt.show()

# Cyg X-1 のデータを使用
period_days = 5.6  # Orbital period [days]
M_bh = 21.2 * M_sun  # Black hole mass [kg]
M_star = 40.6 * M_sun  # Companion star mass [kg]
R_star = 22.3 * SOLAR_RADIUS  # Companion star radius [m]

# 平均距離を計算
distance_cygx1 = calculate_mean_distance(M_bh, M_star, period_days)

# プロットの実行
plot_binary_system_with_velocity(M_bh, M_star, R_star, distance_cygx1, mode="star_wind", axis_symmetric=False)  # 軸対称星風
plot_binary_system_with_velocity(M_bh, M_star, R_star, distance_cygx1, mode="towards_bh", axis_symmetric=False)  # 軸対称視線方向
plot_binary_system_with_velocity(M_bh, M_star, R_star, distance_cygx1, mode="perpendicular_bh", axis_symmetric=False)  # 軸対称垂直方向
