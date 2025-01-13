#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

# プロットのパラメータを設定します
params = {
    'xtick.labelsize': 14,  # x軸目盛りのフォントサイズ
    'ytick.labelsize': 14,  # y軸目盛りのフォントサイズ
    'legend.fontsize': 11,   # 凡例のフォントサイズ
    'axes.labelsize': 14  # xlabel, ylabel のフォントサイズを変更
}
plt.rcParams['font.family'] = 'serif'  # フォントファミリを設定します
plt.rcParams.update(params)

# テキストファイルのパス
file_path_afterdip = 'fit_summary_Hp_afterdip.csv'
file_path_beforedip = 'fit_summary_Hp_beforedip.csv'
file_path_timeave = 'fit_summary_Hp_timeave_epi2.csv'
file_path_timeaveorg = 'fit_summary_Hp_timeave_epi2_nocor.csv'

# データを読み込んで適切に処理する関数
def load_data(file_path):
    data = []
    pixels = []
    xfwhm = []    
    xfwhme = []    
    xgain = []
    xgaine = []
    with open(file_path, 'r') as file:
        for line in file:
            # カンマで区切り、必要な部分だけ抽出
            parts = line.strip().split(',')
            if len(parts) > 3:
                # 3番目のカラム（数値データ）を取得（例：99131.2069）
                try:
                    pixels.append(int(parts[0]))
                    xfwhm.append(float(parts[8]))
                    xfwhme.append(float(parts[9]))
                    xgain.append(float(parts[5]))
                    xgaine.append(float(parts[6]))

                except ValueError:
                    pass  # 無効な値はスキップ
    return np.array(pixels), np.array(xfwhm), np.array(xfwhme), np.array(xgain), np.array(xgaine)

# データを読み込む
pixels_timeave, xfwhm_timeave, xfwhme_timeave, xgain_timeave, xgaine_timeave = load_data(file_path_timeave)
pixels_timeaveorg, xfwhm_timeaveorg, xfwhme_timeaveorg, xgain_timeaveorg, xgaine_timeaveorg = load_data(file_path_timeaveorg)
pixels_afterdip, xfwhm_afterdip, xfwhme_afterdip, xgain_afterdip, xgaine_afterdip = load_data(file_path_afterdip)
pixels_beforedip, xfwhm_beforedip, xfwhme_beforedip, xgain_beforedip, xgaine_beforedip = load_data(file_path_beforedip)


show=True

fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True, sharey=False)
axes[0].set_ylabel(r"$\Delta E$ (eV)")
cut_eachpixel = pixels_afterdip >= -1

errorbar = axes[0].errorbar(pixels_beforedip[cut_eachpixel]-0.25, xfwhm_beforedip[cut_eachpixel], yerr=xfwhme_beforedip[cut_eachpixel], \
                                  fmt='o', color='pink', label = "phase < 0.9", ms=3)
for line in errorbar[2]: # errorbar[2] は誤差棒の Line2D オブジェクト
    line.set_alpha(0.3)  # 誤差棒の透明度を0.3に設

errorbar = axes[0].errorbar(pixels_afterdip[cut_eachpixel]+0.25, xfwhm_afterdip[cut_eachpixel], yerr=xfwhme_afterdip[cut_eachpixel], \
                                  fmt='o', color='skyblue', label = "phase > 0.9", ms=3)
for line in errorbar[2]: # errorbar[2] は誤差棒の Line2D オブジェクト
    line.set_alpha(0.3)  # 誤差棒の透明度を0.3に設

errorbar = axes[0].errorbar(pixels_timeaveorg[cut_eachpixel]-0.1, xfwhm_timeaveorg[cut_eachpixel], yerr=xfwhme_timeaveorg[cut_eachpixel], \
                                  fmt='o', color='gold', label = "entire obs (default)", ms=3)
for line in errorbar[2]: # errorbar[2] は誤差棒の Line2D オブジェクト
    line.set_alpha(0.3)  # 誤差棒の透明度を0.3に設

errorbar = axes[0].errorbar(pixels_timeave[cut_eachpixel], xfwhm_timeave[cut_eachpixel], yerr=xfwhme_timeave[cut_eachpixel], \
                                  fmt='o', color='black', label = "entire obs (reprocessed)", ms=3)
for line in errorbar[2]: # errorbar[2] は誤差棒の Line2D オブジェクト
    line.set_alpha(0.3)  # 誤差棒の透明度を0.3に設


axes[0].legend(numpoints=1, frameon=True, loc="best")

axes[0].grid(linestyle='dotted',alpha=0.1)
axes[0].set_ylim(0,6)

axes[1].set_ylabel(r"E$_{\rm{shift}}$(eV) at Mn K$\alpha$")
ene5900 = 5900. # eV
axes[1].set_xlabel("Pixels")
cut_eachpixel = pixels_afterdip >= -1


errorbar = axes[1].errorbar(pixels_beforedip[cut_eachpixel]-0.25, (xgain_beforedip[cut_eachpixel] - 1.0) * ene5900, \
              yerr=xgaine_beforedip[cut_eachpixel]* ene5900, fmt='o', color='pink', laabel = None, ms=3)
for line in errorbar[2]: # errorbar[2] は誤差棒の Line2D オブジェクト
    line.set_alpha(0.3)  # 誤差棒の透明度を0.3に設

errorbar = axes[1].errorbar(pixels_afterdip[cut_eachpixel]+0.25, (xgain_afterdip[cut_eachpixel] - 1.0) * ene5900, \
              yerr=xgaine_afterdip[cut_eachpixel]* ene5900, fmt='o', color='skyblue', laabel = None, ms=3)
for line in errorbar[2]: # errorbar[2] は誤差棒の Line2D オブジェクト
    line.set_alpha(0.3)  # 誤差棒の透明度を0.3に設

errorbar = axes[1].errorbar(pixels_timeaveorg[cut_eachpixel]-0.1, (xgain_timeaveorg[cut_eachpixel] - 1.0) * ene5900, \
              yerr=xgaine_timeaveorg[cut_eachpixel]* ene5900, fmt='o', color='gold', laabel = None, ms=3)
for line in errorbar[2]: # errorbar[2] は誤差棒の Line2D オブジェクト
    line.set_alpha(0.3)  # 誤差棒の透明度を0.3に設

errorbar = axes[1].errorbar(pixels_timeave[cut_eachpixel], (xgain_timeave[cut_eachpixel] - 1.0) * ene5900, \
              yerr=xgaine_timeave[cut_eachpixel]* ene5900, fmt='o', color='black', laabel = None, ms=3)
for line in errorbar[2]: # errorbar[2] は誤差棒の Line2D オブジェクト
    line.set_alpha(0.3)  # 誤差棒の透明度を0.3に設


axes[1].grid(linestyle='dotted',alpha=0.1)
axes[1].set_ylim(-0.5, 0.5)
sx = -1
sy = 0.1
plt.annotate('', xy=(sx, sy), xytext=(sx, sy-0.25),
                 arrowprops=dict(arrowstyle="->", color='gray'), fontsize=10)
plt.text(sx-0.5, sy-0.35, 'all pixel except for cal', fontsize=10, color='gray')

plt.tight_layout()

plt.savefig("fit_1st_paper_comp_MnKa.png")
plt.savefig("fit_1st_paper_comp_MnKa.eps")

if show:
    plt.show()
