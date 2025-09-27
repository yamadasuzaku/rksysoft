#!/usr/bin/env python 

import numpy as np
import matplotlib.pyplot as plt
# Plotting Configuration
plt.rcParams['font.family'] = 'serif'

from scipy.special import digamma
from scipy.optimize import brentq

# ==============================
# 基本パラメータ（必要に応じて変更）
# ==============================
Tc1    = 1.2     # [K] 超伝導層(単体)の Tc
Theta1 = 100.0   # [K] デバイ温度（式(1)のログに Θ1/Tc1 で入る）
TC_MIN_DEFAULT = 1e-8  # ルート探索の下限温度（mK～μKが必要なら 1e-12 などに下げてOK）

# ==============================
# Kubo 式(1)(2) の実装（digamma厳密）
# ==============================
def rhs_log_term(alpha, beta):
    """ A(α,β) = α/(1+α) * ln( 1 + (1+α)/((Θ1/Tc1)*α*β) ) """
    return (alpha/(1.0+alpha)) * np.log(1.0 + (1.0+alpha)/((Theta1/Tc1)*alpha*beta))

def u_exact(Tc, alpha, beta):
    """ u(Tc) = ψ(1/2 + (1+α)/(2π (Tc/Tc1) α β)) - ψ(1/2) """
    x = 0.5 + (1.0+alpha) / (2.0*np.pi * (Tc/Tc1) * alpha * beta)
    return digamma(x) - digamma(0.5)

def f_equation(Tc, alpha, beta):
    """
    f(T) = ln(T/Tc1) - [ α/(1+α) * { ln( 1 + (1+α)/((Θ1/Tc1) α β) ) - u(T) } ]
    これが 0 になる Tc を求める
    """
    return np.log(Tc/Tc1) - (alpha/(1.0+alpha)) * ( np.log(1.0 + (1.0+alpha)/((Theta1/Tc1)*alpha*beta)) - u_exact(Tc, alpha, beta) )

def _find_bracket(f, T_min, T_max, n_points=300, expand_iters=3,
                  expand_low=10.0, expand_high=10.0,
                  debug=False, debug_max_prints=300):
    """ 符号反転区間を探す（デバッグ出力対応） """
    prints = 0

    def scan_interval(a, b):
        nonlocal prints
        # 低温は幾何刻み、高温は線形刻み
        mid = max(a*1e3, min(b, 1e-3))
        if mid <= a:
            grid = np.linspace(a, b, n_points)
        else:
            n1 = max(10, n_points//4)
            n2 = max(10, n_points - n1)
            g1 = np.geomspace(max(a,1e-16), max(mid,a*10), n1)
            g2 = np.linspace(max(mid,a), b, n2)
            grid = np.unique(np.concatenate([g1, g2]))

        prev_T = grid[0]
        prev_f = f(prev_T)
        if debug and prints < debug_max_prints:
            print(f"[scan] T={prev_T:.6e}  f={prev_f:+.6e}"); prints += 1

        for T in grid[1:]:
            val = f(T)
            if debug and prints < debug_max_prints:
                print(f"[scan] T={T:.6e}  f={val:+.6e}"); prints += 1
            if np.isfinite(prev_f) and np.isfinite(val) and prev_f*val < 0.0:
                if debug:
                    print(f"[bracket found] [{prev_T:.6e}, {T:.6e}]  "
                          f"f(prev)={prev_f:+.6e}, f(now)={val:+.6e}")
                return prev_T, T
            prev_T, prev_f = T, val
        return None, None

    a, b = T_min, T_max
    for k in range(expand_iters+1):
        if debug:
            print(f"\n=== bracket scan iter {k}  range=[{a:.3e}, {b:.3e}] ===")
        x0, x1 = scan_interval(a, b)
        if x0 is not None:
            return x0, x1
        a = max(a/expand_low, 1e-16)  # 下側拡張
        b = b*expand_high              # 上側拡張

    if debug:
        print("[no bracket] no sign change detected in scans.")
    return None, None

def solve_Tc(alpha, beta, T_low=TC_MIN_DEFAULT, T_high=None,
             debug=False, debug_max_prints=300):
    """
    厳密 digamma で f(T)=0 を解く Tc（K）を返す。
    debug=True でブラケット探索と brentq の進捗を print。
    """
    if T_high is None:
        T_high = Tc1  # まず Tc <= Tc1 を仮定

    f = lambda T: f_equation(T, alpha, beta)

    if debug:
        A = rhs_log_term(alpha, beta)
        print(f"[params] alpha={alpha}, beta={beta},  Tc1={Tc1}, Theta1={Theta1}")
        print(f"[A(α,β)] = {A:+.6e}  (log term)")
        print(f"[endpoints] T_low={T_low:.3e}, f_low={f(T_low):+.6e}")
        print(f"[endpoints] T_high={T_high:.3e}, f_high={f(T_high):+.6e}")

    a, b = _find_bracket(f, T_low, T_high, debug=debug, debug_max_prints=debug_max_prints)
    if a is None:
        if debug:
            print("\n[retry] expanding search range aggressively...")
        a2, b2 = _find_bracket(f, max(T_low/1e6, 1e-16), T_high*1e6, n_points=1000,
                               expand_iters=2, expand_low=100.0, expand_high=100.0,
                               debug=debug, debug_max_prints=debug_max_prints)
        if a2 is None:
            if debug:
                print("[solve_Tc] failed: no sign change -> root likely does not exist "
                      "for these parameters.")
            return np.nan
        a, b = a2, b2

    if debug:
        print(f"\n[brentq] start on [{a:.6e}, {b:.6e}] ...")
    try:
        Tc = brentq(f, a, b, maxiter=400, xtol=1e-12, rtol=1e-12)
        if debug:
            print(f"[brentq] converged: Tc={Tc:.9e} K, f(Tc)={f(Tc):+.3e}")
        return Tc
    except Exception as e:
        if debug:
            print(f"[brentq] error: {e}")
        return np.nan

# ==============================
# 計算ユーティリティ（描画と分離）
# ==============================
def compute_tc_over_tc1_for_alphas(betas, alpha_min=0.1, alpha_max=5.0, n_alpha=40, debug=False):
    """
    複数 β について α を掃引し、Tc/Tc1 を返す。
    Returns: alphas (1D), results (dict: beta -> 1D array)
    """
    alphas = np.linspace(alpha_min, alpha_max, n_alpha)
    results = {}
    for beta in np.atleast_1d(betas):
        vals = []
        for a in alphas:
            Tc = solve_Tc(a, beta, debug=debug)  # 必要に応じて debug=True
            vals.append(Tc/Tc1 if np.isfinite(Tc) else np.nan)
        results[float(beta)] = np.array(vals)
    return alphas, results

def compute_tc_over_tc1_for_betas(alphas, beta_min=1e-2, beta_max=1e2, n_beta=40, debug=False):
    """
    複数 α について β を掃引し、Tc/Tc1 を返す（β は対数スケール）。
    Returns: betas (1D), results (dict: alpha -> 1D array)
    """
    betas = np.logspace(np.log10(beta_min), np.log10(beta_max), n_beta)
    results = {}
    for alpha in np.atleast_1d(alphas):
        vals = []
        for b in betas:
            Tc = solve_Tc(alpha, b, debug=debug)
            vals.append(Tc/Tc1 if np.isfinite(Tc) else np.nan)
        results[float(alpha)] = np.array(vals)
    return betas, results

# ==============================
# 描画関数
# ==============================
def plot_alpha_sweep(betas, alpha_min=0.1, alpha_max=5.0, n_alpha=40, debug=False):
    alphas, res = compute_tc_over_tc1_for_alphas(
        betas, alpha_min, alpha_max, n_alpha, debug=debug
    )

    # 右に凡例を出すので少し横長に
    fig, ax = plt.subplots(figsize=(7.5, 4))

    for beta, vals in res.items():
        ax.plot(alphas, vals, '-', label=rf'$\beta={beta}$')

    ax.set_xlabel(r'$\alpha = N_2 d_2 / (N_1 d_1)$')
    ax.set_ylabel(r'$T_c/T_{c1}$')
    ax.set_title('Tc vs α (Kubo Eq.(1), exact digamma)')
    ax.grid(True, which='both', alpha=0.5)

    # 凡例を枠外（右）へ
    ax.legend(
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),   # 軸の右外 2%
        frameon=True,
        title=r'$\beta$'
    )

    # 右側に凡例スペースを確保
    fig.tight_layout(rect=[0, 0, 0.82, 1])

    outfig = "alpha_sweep.png"
    fig.savefig(outfig, bbox_inches='tight', dpi=200)  # 切れ防止
    print(f"{outfig} is created.")

    plt.show()


def plot_beta_sweep(alphas, beta_min=1e-2, beta_max=1e2, n_beta=40, debug=False):
    betas, res = compute_tc_over_tc1_for_betas(alphas, beta_min, beta_max, n_beta, debug=debug)

    # 右に凡例を出すので少し横長に
    fig, ax = plt.subplots(figsize=(7.5, 4))

    for alpha, vals in res.items():
        ax.semilogx(betas, vals, '-', label=rf'$\alpha={alpha}$')

    ax.set_xlabel(r'$\beta$ (interface parameter)')
    ax.set_ylabel(r'$T_c/T_{c1}$')
    ax.set_title('Tc vs β (Kubo Eq.(1), exact digamma)')
    ax.grid(True, which='both', alpha=0.5)

    # 枠外（右）に凡例
    ax.legend(
        loc='center left',            # 左寄せで
        bbox_to_anchor=(1.02, 0.5),   # 軸の右外 2% に配置
        frameon=True,
        title=r'$\alpha$'
    )

    # 右側に余白を残す（凡例スペース）
    fig.tight_layout(rect=[0, 0, 0.82, 1])

    outfig = "beta_sweep.png"
    fig.savefig(outfig, bbox_inches='tight', dpi=200)  # 切れ防止
    print(f"{outfig} is created.")

    plt.show()

# ==============================
# 使 用 例
# ==============================
# 1) α 掃引（β を複数固定）
#   極端に小さい β では Tc が mK～μK に落ちるため、下限 TC_MIN_DEFAULT を 1e-10 などに下げると良いです。
plot_alpha_sweep(betas=[0.1, 0.5, 1.0, 1.5, 2.0, 2.5], alpha_min=0.01, alpha_max=5.0, n_alpha=80, debug=False)

# 2) β 掃引（α を複数固定）
plot_beta_sweep(alphas=[0.1, 0.5, 1.0, 2.0, 3.0], beta_min=1e-4, beta_max=1e2, n_beta=80, debug=False)
