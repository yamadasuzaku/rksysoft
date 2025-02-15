#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
import math
import argparse

# 定数の設定
G = 6.67430e-8  # 万有引力定数 (cm^3/g/s^2)
m_p = 1.6726e-24  # 陽子質量 (g)
c = 3e10  # 光速 (cm/s)
sigma_T = 6.652e-25  # トムソン散乱断面積 (cm^2)
solar_mass = 1.989e33  # 太陽質量 (g)

# 関数：光度とエディントン光度の比を計算
def calculate_luminosity_ratio(flux, distance_kpc, mass_solar):
    # 距離の単位変換
    distance_cm = distance_kpc * 3.086e21  # kpc → cm

    # 光度の計算
    luminosity = 4 * math.pi * distance_cm**2 * flux

    # エディントン光度の計算
    mass = mass_solar * solar_mass  # 質量 (g)
    L_edd = (4 * math.pi * G * mass * m_p * c) / sigma_T  # エディントン光度

    # 光度とエディントン光度の比を計算
    luminosity_ratio = luminosity / L_edd
    return luminosity, luminosity_ratio, L_edd

# argparse を使用して引数を処理
def main():
    parser = argparse.ArgumentParser(description='Calculate luminosity and Eddington luminosity ratio.')
    parser.add_argument('flux', type=float, help='Flux in erg/cm^2/s')
    parser.add_argument('mass', type=float, help='Mass in solar masses')
    parser.add_argument('distance', type=float, help='Distance in kpc')
    parser.add_argument('--flux_min', type=float, default=-10, help='Minimum flux range (default: -10)')
    parser.add_argument('--flux_max', type=float, default=-6, help='Maximum flux range (default: -6)')
    parser.add_argument('--outfile', type=str, default="mass_lumi_limiedd.png", help='output file name')

    
    args = parser.parse_args()

    # 引数から値を取得
    flux = args.flux
    mass_solar = args.mass
    distance_kpc = args.distance
    flux_min = args.flux_min
    flux_max = args.flux_max
    outfile = args.outfile

    # 光度とエディントン光度の比を計算
    user_luminosity, user_luminosity_ratio, user_L_edd = calculate_luminosity_ratio(flux, distance_kpc, mass_solar)

    # 結果を表示
    print(f"Calculated luminosity: {user_luminosity:.3e} erg/s")
    print(f"Luminosity to Eddington luminosity ratio: {user_luminosity_ratio:.3e}")
    print(f"Eddington luminosity for input mass: {user_L_edd:.3e} erg/s")

    # 可視化
    flux_values = np.logspace(flux_min, flux_max, 100)  # フラックスの範囲を変更可能に
    luminosities = []
    luminosity_ratios = []

    for flux_val in flux_values:
        luminosity, luminosity_ratio, _ = calculate_luminosity_ratio(flux_val, distance_kpc, mass_solar)
        luminosities.append(luminosity)
        luminosity_ratios.append(luminosity_ratio)

    # 可視化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # 光度のプロット
    ax1.plot(flux_values, luminosities, label='Luminosity (erg/s)', color='blue')
    ax1.plot(flux, user_luminosity, "o", label='(USER) Luminosity (erg/s)', color='red')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel(r'Flux (erg/cm$^2$/s)')
    ax1.set_ylabel('Luminosity (erg/s)')
    ax1.set_title('Luminosity vs Flux')
    ax1.legend()

    # エディントン光度比のプロット
    ax2.plot(flux_values, luminosity_ratios, label=r'$L / L_{\rm{Edd}}$', color='blue')
    ax2.plot(flux, user_luminosity_ratio, "o", label=r'(USER) $L / L_{\rm{Edd}}$', color='red')

    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel(r'Flux (erg/cm$^2$/s)')
    ax2.set_ylabel(r'$L / L_{\rm{Ledd}}$')
    ax2.set_title(r'Luminosity to Eddington Luminosity Ratio')
    ax2.legend()

    # 上部余白を調整し、テキストを配置
    fig.subplots_adjust(top=0.8)  # 上部の余白を大きく設定

    # 入力パラメータのテキスト表示
    input_text_str = (f"Input Parameters:\n"
                      f"Flux: {flux} erg/cm²/s\n"
                      f"Mass: {mass_solar}" r"$M_{\odot}$\n"
                      f"Distance: {distance_kpc} kpc")

    # 計算結果のテキスト表示
    output_text_str = (f"Calculated Results:\n"
                       f"Luminosity: {user_luminosity:.3e} erg/s\n"
                       f"Eddington Luminosity: {user_L_edd:.3e} erg/s\n"
                       f"Luminosity / Eddington Luminosity Ratio: {user_luminosity_ratio:.3e}")

    # テキストを図の上部に表示
    plt.figtext(0.05, 0.87, input_text_str, transform=fig.transFigure, fontsize=8,
                verticalalignment='bottom', horizontalalignment='left', color='black',
                bbox=dict(facecolor='white', alpha=0.8))

    plt.figtext(0.55, 0.87, output_text_str, transform=fig.transFigure, fontsize=8,
                verticalalignment='bottom', horizontalalignment='left', color='black',
                bbox=dict(facecolor='white', alpha=0.8))

#    plt.tight_layout()
    plt.savefig(outfile)
    plt.show()

if __name__ == '__main__':
    main()
