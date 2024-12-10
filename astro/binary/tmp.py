#!/usr/bin/env python 

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'

# グローバル変数
INCLINATION=30 # [deg] 

# パラメータ
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

# 補間関数
def interpolate(value_0, value_20, theta):
    return value_0 + (value_20 - value_0) * (theta / 20)**2

# 星風の速度を計算
def v_wind(r, theta, stellar_radius):
    params = parameters[stellar_radius]
    r_star_0, r_star_20 = params[0]["r_star"], params[20]["r_star"]
    a_0, a_20 = params[0]["a"], params[20]["a"]
    v_inf_0, v_inf_20 = params[0]["v_inf"], params[20]["v_inf"]

    r_star_theta = interpolate(r_star_0, r_star_20, theta)
    a_theta = interpolate(a_0, a_20, theta)
    v_inf_theta = interpolate(v_inf_0, v_inf_20, theta)

    return v_inf_theta * (1 - r_star_theta / r)**a_theta

# 星風の密度を計算
def rho_wind(r, theta, stellar_radius):
    params = parameters[stellar_radius]
    r_star_0, r_star_20 = params[0]["r_star"], params[20]["r_star"]
    rho_0_0, rho_0_20 = params[0]["rho_0"], params[20]["rho_0"]
    a_0, a_20 = params[0]["a"], params[20]["a"]

    r_star_theta = interpolate(r_star_0, r_star_20, theta)
    rho_0_theta = interpolate(rho_0_0, rho_0_20, theta)
    a_theta = interpolate(a_0, a_20, theta)

    numerator = (r_star_theta / r)**2 * rho_0_theta
    denominator = (1 - (r_star_theta / r))**a_theta

    return numerator / denominator

# 可視化
def plot_v_rho(stellar_radius):
    solar_radius = 6.96e10  # 太陽半径 [cm]
    stellar_radius_in_solar = 22  # 星の半径は太陽半径の22倍
    r_values = np.linspace(solar_radius, 50 * solar_radius, 500)  # 半径 1-50 太陽半径
    theta_values = [0, 5, 10, 15, 20]  # θの値 [deg]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    # v_windのプロット
    ax1 = axes[0]
    ax2 = ax1.secondary_xaxis('top', functions=(lambda x: x / stellar_radius_in_solar, 
                                                lambda x: x * stellar_radius_in_solar))
    ax2.set_xlabel(r"$R$ [$R_{star}$]")

    for theta in theta_values:
        v_values = [v_wind(r, theta, stellar_radius) for r in r_values]
        v_inf_theta = interpolate(
            parameters[stellar_radius][0]["v_inf"],
            parameters[stellar_radius][20]["v_inf"],
            theta,
        )
        # プロットと水平線の色を一致
        line, = ax1.plot(r_values / solar_radius, v_values, label=f"Theta = {theta}°")
        ax1.axhline(y=v_inf_theta, color=line.get_color(), linestyle="--", label=f"v_inf ({theta}°)")

    ax1.set_xlabel(r"$R$ [$R_{\odot}$]")
    ax1.set_ylabel(r"$v_{wind}$ [km/s]")
    ax1.set_title(f"Stellar Radius = {stellar_radius} d (velocity), " + r"$R_{star}$=" + f"{stellar_radius_in_solar}" + r"$R_{\odot}$")
    ax1.legend()
    ax1.grid()

# rho_windのプロット（質量密度と粒子数密度の両方をプロット）
    ax3 = axes[1]
    ax4 = ax3.secondary_xaxis('top', functions=(lambda x: x / stellar_radius_in_solar, 
                                                lambda x: x * stellar_radius_in_solar))
    ax4.set_xlabel(r"$R$ [$R_{star}$]")

    # 右側に第2のy軸を追加
    m_H = 1.67e-24  # 水素原子の質量 [g]
    mu = 1  # 平均分子量
    ax5 = ax3.secondary_yaxis('right', functions=(lambda rho: rho / (mu * m_H),
                                                  lambda n: n * (mu * m_H)))
    ax5.set_ylabel(r"$n_{wind}$ [cm$^{-3}$]")  # 粒子数密度のラベル

    for theta in theta_values:
        rho_values = [rho_wind(r, theta, stellar_radius) for r in r_values]
        ax3.plot(r_values / solar_radius, rho_values, label=f"Theta = {theta}°")

    ax3.set_xlabel(r"$R$ [$R_{\odot}$]")
    ax3.set_ylabel(r"$\rho_{wind}$ [g/cm³]")
    ax3.set_title(f"Stellar Radius = {stellar_radius} d (density)")
    ax3.set_yscale("log")  # 密度は対数スケールでプロット
    ax3.legend()
    ax3.grid()

    plt.tight_layout()
    plt.savefig(f"steller_profile_{stellar_radius}.png",bbox_inches='tight')
    plt.show()

    # plot for projected wind velocities 
    r_values_forproj = np.linspace(solar_radius * 25, 50 * solar_radius, 5)  # 半径 1-50 太陽半径


    fig, axes = plt.subplots(len(theta_values), 1, figsize=(10, 8), sharex=True)
    fig.subplots_adjust(right=0.8)  # 右に余白を追加
    fig.suptitle("Projected Velocities", fontsize=12)

    for i, theta in enumerate(theta_values):
#        fig, axes = plt.subplots(1, 1, figsize=(14, 6))
        v_values = [v_wind(r, theta, stellar_radius) for r in r_values_forproj]

        for _v, _r in zip(v_values, r_values_forproj):
            phase = np.linspace(0,4*np.pi,100) # 2 pi x 2  
            view_angle = np.cos(phase) * INCLINATION # [deg]
            v_prog = _v * np.sin(np.deg2rad(theta - view_angle))

            axes[i].plot(phase/(2*np.pi), v_prog, label=fr"$\theta$ = {theta}°, $R$ [$R$sun] = {_r / solar_radius:2.1f}, $V$wind = {_v:2.1f} km/s")
            axes[i].set_ylabel(r"$V_{proj}$[km/s]")
            axes[i].set_ylim(-1000,1000)
            axes[i].legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=10)  # 右余白に配置
            axes[i].grid(alpha=0.3)

    axes[-1].set_xlabel("Orbital phase")  # 最下部のパネルにxlabelを設定
    plt.tight_layout()
    plt.savefig(f"steller_profile_{stellar_radius}_proj_theta{theta:d}.png",bbox_inches='tight')
    plt.show()

# 実行
if __name__ == "__main__":
    stellar_radius = 0.5  # (単位は連星間距離 d, 角度0のBH方向) モデルの選択（0.5 または 0.4）
    plot_v_rho(stellar_radius)

    stellar_radius = 0.4  # (単位は連星間距離 d, 角度0のBH方向) 別モデルで再実行
    plot_v_rho(stellar_radius)
