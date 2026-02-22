import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------
# 物理定数と基本設定
# ----------------------------------------
G = 6.6743e-11         # gravitational constant [m^3 kg^-1 s^-2]
c = 2.99792458e8       # speed of light [m/s]
M_sun = 1.98847e30     # solar mass [kg]

# ブラックホール質量の例 (10 Msun)
M = 10.0 * M_sun

# 重力半径 Rg = GM/c^2
Rg = G * M / c**2

# ----------------------------------------
# ユーティリティ関数
# ----------------------------------------
def t_K(r):
    """
    ケプラー周期 t_K [s]
    r: 無次元半径 R/Rg
    """
    R = r * Rg
    Omega_K = np.sqrt(G * M / R**3)
    return 2.0 * np.pi / Omega_K

def t_acc(r, alpha, H_over_R):
    """
    降着時間 t_acc [s]
    r: 無次元半径 R/Rg
    alpha: alpha parameter
    H_over_R: H/R
    """
    return (1.0 / alpha) * (1.0 / H_over_R**2) * t_K(r)

def v_r_over_vK(alpha, H_over_R):
    """
    radial velocity / Keplerian velocity
    v_r / v_K = alpha * (H/R)^2
    """
    return alpha * H_over_R**2

# 共通パラメータ
alpha = 0.1

# ========================================
# Figure 1: t_acc / t_K vs H/R
# ========================================
H_over_R_vals = np.logspace(-3.0, 0, 300)  # 10^-3 ~ 1
tacc_over_tk = (1.0 / alpha) * (1.0 / H_over_R_vals**2)

fig1, ax1 = plt.subplots(figsize=(6, 4))

ax1.loglog(H_over_R_vals, tacc_over_tk)

# thin / thick regime のシェーディング
ax1.axvspan(1e-3, 0.1, alpha=0.1)  # thin disk regime
ax1.axvspan(0.3, 1.0, alpha=0.1)   # thick/ADAF regime

# thin disk example
x_thin = 0.05
y_thin = (1.0 / alpha) * (1.0 / x_thin**2)
ax1.scatter([x_thin], [y_thin], marker="o")
ax1.annotate(
    "thin disk\n\nH/R=0.05\n",
    xy=(x_thin, y_thin),
    xytext=(0.02, y_thin * 6.0),
    arrowprops=dict(arrowstyle="->"),
    ha="left",
)

# ADAF example
x_adaf = 1.0
y_adaf = (1.0 / alpha) * (1.0 / x_adaf**2)
ax1.scatter([x_adaf], [y_adaf], marker="s")
ax1.annotate(
    "ADAF\n\nH/R=1.0\n",
    xy=(x_adaf, y_adaf),
    xytext=(0.4, y_adaf * 6.0),
    arrowprops=dict(arrowstyle="->"),
    ha="left",
)

# slope -2 のガイドライン (参考)
H_ref = 0.1
y_ref = (1.0 / alpha) * (1.0 / H_ref**2)
guide = y_ref * (H_over_R_vals / H_ref)**(-2.0)
ax1.loglog(H_over_R_vals, guide, linestyle="--")

ax1.text(0.2, y_ref * 0.3, "slope -2", rotation=-40)

# ラベルとタイトル（ASCIIのみ）
ax1.set_xlabel("H/R")
ax1.set_ylabel("t_acc / t_K")
ax1.set_title("Accretion timescale ratio vs H_over_R (alpha=0.1)")
ax1.grid(True, which="both", ls="--", alpha=0.5)

# regime のテキスト
ax1.text(2e-3, 5e5, "thin disk regime", fontsize=9)
ax1.text(0.35, 5e1, "thick / ADAF regime", fontsize=9)

fig1.tight_layout()
fig1.savefig("fig1_tacc_over_tk_vs_HoverR.png", dpi=200)

# ========================================
# Figure 2: t_acc vs radius (thin vs ADAF)
# ========================================
r_vals = np.logspace(1, 3, 300)  # 10 - 1000 Rg

H_over_R_thin = 0.05
H_over_R_ADAF = 1.0

tacc_thin = t_acc(r_vals, alpha, H_over_R_thin)
tacc_ADAF = t_acc(r_vals, alpha, H_over_R_ADAF)

fig2, ax2 = plt.subplots(figsize=(6, 4))

ax2.loglog(r_vals, tacc_thin, label="thin disk (H/R=0.05)")
ax2.loglog(r_vals, tacc_ADAF, label="ADAF (H/R=1.0)")

# 具体例: r = 100 Rg
r_example = 100.0
t_thin_100 = t_acc(r_example, alpha, H_over_R_thin)
t_ADAF_100 = t_acc(r_example, alpha, H_over_R_ADAF)

ax2.scatter([r_example], [t_thin_100])
ax2.scatter([r_example], [t_ADAF_100])

ax2.annotate(
    f"thin: t_acc ~ {t_thin_100:.1f} s",
    xy=(r_example, t_thin_100),
    xytext=(60, t_thin_100 * 5),
    arrowprops=dict(arrowstyle="->"),
)

ax2.annotate(
    f"ADAF: t_acc ~ {t_ADAF_100:.2f} s",
    xy=(r_example, t_ADAF_100),
    xytext=(120, t_ADAF_100 / 5),
    arrowprops=dict(arrowstyle="->"),
)

ax2.set_xlabel("R / Rg")
ax2.set_ylabel("t_acc [s]")
ax2.set_title("Accretion timescale vs radius (M=10Msun, alpha=0.1)")
ax2.grid(True, which="both", ls="--", alpha=0.5)
ax2.legend()

fig2.tight_layout()
fig2.savefig("fig2_tacc_vs_r_thin_vs_ADAF.png", dpi=200)

# ========================================
# Figure 3: v_r / v_K vs H/R
# ========================================
H_over_R_vals2 = np.logspace(-3.0, 0, 300)
vr_over_vK_vals = v_r_over_vK(alpha, H_over_R_vals2)

fig3, ax3 = plt.subplots(figsize=(6, 4))

ax3.loglog(H_over_R_vals2, vr_over_vK_vals)

# thin / thick regime のシェーディング
ax3.axvspan(1e-3, 0.1, alpha=0.1)
ax3.axvspan(0.3, 1.0, alpha=0.1)

# example points
x_thin2 = 0.05
y_thin2 = v_r_over_vK(alpha, x_thin2)
ax3.scatter([x_thin2], [y_thin2], marker="o")
ax3.annotate(
    f"thin disk\nv_r/v_K ~ {y_thin2:.2e}\n",
    xy=(x_thin2, y_thin2),
    xytext=(0.01, y_thin2 * 2),
    arrowprops=dict(arrowstyle="->"),
    ha="left",
)

x_adaf2 = 1.0
y_adaf2 = v_r_over_vK(alpha, x_adaf2)
ax3.scatter([x_adaf2], [y_adaf2], marker="s")
ax3.annotate(
    f"ADAF v_r/v_K ~ {y_adaf2:.1f}\n",
    xy=(x_adaf2, y_adaf2),
    xytext=(0.1, y_adaf2 * 0.2),
    arrowprops=dict(arrowstyle="->"),
    ha="left",
)

ax3.set_xlabel("H/R")
ax3.set_ylabel("v_r / v_K")
ax3.set_title("Radial velocity fraction vs H_over_R (alpha=0.1)")
ax3.grid(True, which="both", ls="--", alpha=0.5)

ax3.text(2e-3, 5e-4, "thin disk regime", fontsize=9)
ax3.text(0.35, 0.01, "thick / ADAF regime", fontsize=9)

fig3.tight_layout()
fig3.savefig("fig3_vr_over_vK_vs_HoverR.png", dpi=200)

plt.show()
