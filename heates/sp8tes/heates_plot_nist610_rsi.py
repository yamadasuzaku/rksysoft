#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" plot_nist610.py

このスクリプトは、NIST 610のデータをプロットするためのものです。
歴史:
2019-07-18 ; バージョン 1.0 (移植版)
2024-08-28 ; バージョン 2.0 (pytyon3系に移植)
"""

__author__ = 'Shinya Yamada'
__version__= '2.0'

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, FixedLocator

# プロットのパラメータを設定します
params = {
    'xtick.labelsize': 16,  # x軸目盛りのフォントサイズ
    'ytick.labelsize': 16,  # y軸目盛りのフォントサイズ
    'legend.fontsize': 16   # 凡例のフォントサイズ
}
plt.rcParams['font.family'] = 'serif'  # フォントファミリを設定します
plt.rcParams.update(params)

# データファイルを開きます
print("データファイルを読み込んでいます...")
with open("NIST610-30sec-SDD.txt") as fsdd, open("run0141_energyall.txt") as ftes:
    esdd = []
    hsdd = []
    # SDDデータを読み込みます
    for line in fsdd:
        values = line.strip().split()
        energy = float(values[0])
        counts = float(values[1])
        if 2.0 < energy < 15.0 and counts > 1.0:
            esdd.append(energy * 1e3)  # エネルギーをeVに変換
            hsdd.append(counts * 0.4)  # カウントをスケールします

    etes = []
    htes = []
    # TESデータを読み込みます
    for line in ftes:
        values = line.strip().split(",")
        energy = float(values[0])
        counts = float(values[1])
        if energy > 2000:
            etes.append(energy)
            htes.append(counts)

# データをNumPy配列に変換します
esdd = np.array(esdd)
hsdd = np.array(hsdd)
etes = np.array(etes)
htes = np.array(htes)

#################################################
# カラープロットの作成
print("カラープロットを作成しています...")

corfactor = 7460. / 7458  # Ni Kα2の補正係数

# 両方のプロットを線形スケールで作成します
fig = plt.figure(figsize=(12, 8))

# メインのプロット領域
ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
# インセットプロット領域
ax2 = fig.add_axes([0.17, 0.55, 0.32, 0.36])

# 軸ラベルを設定します
ax1.set_ylabel("Counts", fontsize=16)
ax1.set_xlabel('Energy (eV)', fontsize=16)
ax1.set_xscale('linear')
ax1.set_yscale('linear')

# プロット範囲を設定します
xmin = 4800
xmax = 11800
ixmin = 7300
ixmax = 7700
tmpyv = 1300

ax1.set_xlim(xmin, xmax)
ax1.set_ylim(1, 3800)

# データをプロットします
ax1.errorbar(esdd, hsdd, fmt='b--', alpha=0.4, label="SDD 0.5 min x 0.4", lw=2, ms=1)
ax1.errorbar(etes, htes, fmt='r-', alpha=0.8, label="TES 10 min", lw=1)

# ドット線を描画します
ax1.plot([5800, ixmin], [1800, tmpyv], c="k", ls="dotted", alpha=0.2, lw=1.5)
ax1.plot([7900, ixmax], [1800, tmpyv], c="k", ls="dotted", alpha=0.2, lw=1.5)
ax1.vlines(x=[ixmin, ixmax], ymin=1000, ymax=tmpyv, colors="k", alpha=0.2, linestyles='dotted', lw=1.5)

# 凡例を追加します
ax1.legend(numpoints=1, frameon=False, loc='best')

# 不要な枠線を削除します
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)

# インセットプロットの設定
ax2.set_xlim(ixmin, ixmax)
ax2.set_ylim(0, 2000)
ax2.errorbar(esdd, hsdd, fmt='b--', alpha=0.4, label="SDD 0.5 min x 0.4", lw=2, ms=1)
ax2.errorbar(etes * corfactor, htes, fmt='r-', alpha=0.8, label="TES 10 min", lw=2)

# 注釈を追加します
offset = 10
ax2.text(7460 - offset, 1850, r"Ni $K_{\alpha2}$", rotation=70, color="m", fontsize=12)
ax2.text(7478 - offset, 1850, r"Ni $K_{\alpha1}$", rotation=70, color="m", fontsize=12)
ax2.text(7367.3 - offset, 1850, r"Yb $L_{\alpha2}$", rotation=70, color="g", fontsize=12)
ax2.text(7415.6 - offset, 1850, r"Yb $L_{\alpha1}$", rotation=70, color="g", fontsize=12)
ax2.text(7525.3 - offset, 1850, r"Ho $L_{\beta1}$", rotation=70, color="c", fontsize=12)
ax2.text(7604.9 - offset, 1850, r"Lu $L_{\alpha2}$", rotation=70, color="#a65628", fontsize=12)
ax2.text(7655.5 - offset, 1850, r"Lu $L_{\alpha1}$", rotation=70, color="#a65628", fontsize=12)

ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)

# 画像を保存します
plt.savefig("nist610_rsi_color.png")
print("nist610_rsi_color.png is created.")
