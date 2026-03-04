#!/usr/bin/env python 
# -*- coding: utf-8 -*-

"""
qiita_plot_time_bode_nyquist_argparse.py

目的
----
1次/2次/3次（= 1次×2次 直列）系について、
- ステップ応答（時間領域）
- Bode 線図（周波数領域）
- Nyquist 線図（複素平面）
を描画し、図を「パラメータを含むファイル名」で自動保存する。

この版の特徴
------------
- argparse でパラメータをコマンドラインから変更できる
- scipy.signal.TransferFunction を中心に実装（初学者向けのコメント多め）
- 画像は outdir 以下に自動保存（prefix やパラメータを名前に埋め込み）

使い方（例）
------------
# デフォルト（tau=1, omega0=1, zeta=0.3）
python qiita_plot_time_bode_nyquist_argparse.py --outdir figs

# 2次系をより弱減衰に
python qiita_plot_time_bode_nyquist_argparse.py --zeta 0.1 --outdir figs

# 画像を PNG と PDF で保存し、画面表示はしない
python qiita_plot_time_bode_nyquist_argparse.py --outdir figs --formats png pdf --no-show
"""

from __future__ import annotations

import argparse
import datetime as _dt
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"

from scipy import signal
import os, textwrap, json, math


# ===========================================================
# 1) TransferFunction の作り方（ここが重要）
# ===========================================================
# scipy.signal.TransferFunction(num, den) は連続時間 LTI 系
#   H(s) = (b0*s^m + b1*s^(m-1) + ... + bm) / (a0*s^n + a1*s^(n-1) + ... + an)
# を
#   num = [b0, b1, ..., bm]
#   den = [a0, a1, ..., an]
# という「多項式係数の配列」で表現します。
#
# 例: 1次遅れ H(s)=1/(tau*s+1) なら
#   num=[1], den=[tau, 1]
#
# ※ signal.TransferFunction は「連続時間系」用です。
#   離散時間なら signal.dlti など別クラスがあります。


def tf_first_order(tau: float) -> signal.TransferFunction:
    """
    一次遅れ系
      H1(s) = 1 / (1 + tau*s)

    num=[1], den=[tau, 1]
    """
    num = [1.0]
    den = [tau, 1.0]
    return signal.TransferFunction(num, den)


def tf_second_order(omega0: float, zeta: float) -> signal.TransferFunction:
    """
    二次遅れ系（標準的な2次系）
      H2(s) = omega0^2 / (s^2 + 2*zeta*omega0*s + omega0^2)

    num=[omega0^2], den=[1, 2*zeta*omega0, omega0^2]
    """
    num = [omega0**2]
    den = [1.0, 2.0 * zeta * omega0, omega0**2]
    return signal.TransferFunction(num, den)


def tf_series(sys1: signal.TransferFunction, sys2: signal.TransferFunction) -> signal.TransferFunction:
    """
    直列接続（series）: H(s) = H1(s)*H2(s)

    伝達関数が「多項式の比」なので、
      (N1/D1) * (N2/D2) = (N1*N2) / (D1*D2)
    となり、分子/分母の多項式は「係数配列の畳み込み（polymul）」で作れます。
    """
    num = np.polymul(sys1.num, sys2.num)
    den = np.polymul(sys1.den, sys2.den)
    return signal.TransferFunction(num, den)


def tf_third_order(tau: float, omega0: float, zeta: float) -> signal.TransferFunction:
    """
    三次遅れ系の例（1次×2次を直列）
      H3(s) = H1(s) * H2(s)
    """
    return tf_series(tf_first_order(tau), tf_second_order(omega0, zeta))


# ===========================================================
# 2) 周波数応答 / ステップ応答の計算
# ===========================================================

def freq_response(sys: signal.TransferFunction, w: np.ndarray) -> np.ndarray:
    """
    周波数応答 H(iω) を計算する。

    scipy.signal.freqresp(sys, w) を使うと、
    s=iω を代入した複素周波数応答が返ります。

    戻り値:
      H: complex ndarray, shape=(len(w),)
    """
    w_out, H = signal.freqresp(sys, w=w)
    # w_out は入力 w と同じ（念のため受け取っている）
    return H


def step_response(sys: signal.TransferFunction, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    ステップ応答 y(t) を計算する。

    signal.step(sys, T=t) は
    入力 u(t)=1(t)（単位ステップ）に対する出力 y(t) を返す。
    """
    tt, yy = signal.step(sys, T=t)
    return tt, yy


# ===========================================================
# 3) 保存ファイル名を安全に作る
# ===========================================================

def _sanitize(s: str) -> str:
    """
    ファイル名に使いにくい文字を '_' に置換して安全化する。
    """
    s = s.strip()
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    return s


def build_tag(tau: float, omega0: float, zeta: float, wmin: float, wmax: float, tmax: float) -> str:
    """
    図のファイル名に埋め込むタグを作る（例: tau1_om1_z0p3_w1e-2-1e2_t20）
    """
    def ffloat(x: float) -> str:
        # 0.3 -> 0p3 のようにしてファイル名を見やすくする
        s = f"{x:g}"
        s = s.replace(".", "p").replace("-", "m")
        return s

    tag = f"tau{ffloat(tau)}_om{ffloat(omega0)}_z{ffloat(zeta)}_w{wmin:g}-{wmax:g}_t{tmax:g}"
    return _sanitize(tag)


def save_figure(fig: plt.Figure, outdir: Path, stem: str, formats: list[str], dpi: int) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        ext = ext.lower().lstrip(".")
        outpath = outdir / f"{stem}.{ext}"
        fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
        print(f"[saved] {outpath}")


# ===========================================================
# 4) メイン: 3つの図を作って保存
# ===========================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot step response, Bode, and Nyquist for 1st/2nd/3rd order systems (scipy.signal.TransferFunction)."
    )

    # ---- システムパラメータ ----
    parser.add_argument("--tau", type=float, default=1.0, help="Time constant tau for 1st-order (and 3rd-order).")
    parser.add_argument("--omega0", type=float, default=1.0, help="Natural angular frequency omega0 for 2nd-order (and 3rd-order).")
    parser.add_argument("--zeta", type=float, default=0.3, help="Damping ratio zeta for 2nd-order (and 3rd-order).")

    # ---- 周波数・時間軸 ----
    parser.add_argument("--wmin", type=float, default=1e-2, help="Min angular frequency [rad/s] for Bode/Nyquist.")
    parser.add_argument("--wmax", type=float, default=1e2, help="Max angular frequency [rad/s] for Bode/Nyquist.")
    parser.add_argument("--nw", type=int, default=1000, help="Number of frequency points.")

    parser.add_argument("--tmax", type=float, default=20.0, help="Max time [s] for step response.")
    parser.add_argument("--nt", type=int, default=1000, help="Number of time points for step response.")

    # ---- 出力 ----
    parser.add_argument("--outdir", type=str, default="figs", help="Output directory for figures.")
    parser.add_argument("--prefix", type=str, default="lti_demo", help="Filename prefix for saved figures.")
    parser.add_argument("--formats", nargs="+", default=["png"], help="Figure formats to save: png pdf svg ...")
    parser.add_argument("--dpi", type=int, default=200, help="DPI for raster formats (e.g., png).")

    parser.add_argument("--no-show", action="store_true", help="Do not display figures (useful for batch runs).")

    args = parser.parse_args()

    # ---- 軸の用意 ----
    # logspace の引数は「指数」なので log10(wmin), log10(wmax) を使う
    w = np.logspace(np.log10(args.wmin), np.log10(args.wmax), args.nw)
    t = np.linspace(0.0, args.tmax, args.nt)

    # ---- システム生成（TransferFunction）----
    sys1 = tf_first_order(args.tau)
    sys2 = tf_second_order(args.omega0, args.zeta)
    sys3 = tf_third_order(args.tau, args.omega0, args.zeta)

    systems = {
        "1st order": sys1,
        "2nd order": sys2,
        "3rd order": sys3,
    }

    # ===========================================================
    # 図1: ステップ応答
    # ===========================================================
    fig_step, ax_step = plt.subplots(figsize=(8, 4))

    for label, sys in systems.items():
        tt, yy = step_response(sys, t)
        ax_step.plot(tt, yy, label=label)

    ax_step.set_xlabel("Time t [s]")
    ax_step.set_ylabel("Step response y(t)")
    ax_step.set_title("Step responses (1st / 2nd / 3rd order)")
    ax_step.grid(True, linestyle=":")
    ax_step.legend()

    # ===========================================================
    # 図2: Bode 線図（ゲイン & 位相）
    # ===========================================================
    fig_bode, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    for label, sys in systems.items():
        H = freq_response(sys, w)
        mag_db = 20.0 * np.log10(np.abs(H))
        phase_deg = np.angle(H, deg=True)

        ax_mag.semilogx(w, mag_db, label=label)
        ax_phase.semilogx(w, phase_deg, label=label)

    ax_mag.set_ylabel("Magnitude [dB]")
    ax_mag.grid(True, which="both", linestyle=":")
    ax_mag.legend()

    ax_phase.set_xlabel("Angular frequency ω [rad/s]")
    ax_phase.set_ylabel("Phase [deg]")
    ax_phase.grid(True, which="both", linestyle=":")

    fig_bode.suptitle("Bode plots (using scipy.signal.freqresp)", fontsize=14)

    # ===========================================================
    # 図3: Nyquist 線図（複素平面）
    # ===========================================================
    fig_nyq, ax_nyq = plt.subplots(figsize=(6, 6))

    for label, sys in systems.items():
        H = freq_response(sys, w)
        ax_nyq.plot(H.real, H.imag, label=label)

    ax_nyq.set_xlabel("Re{H(iω)}")
    ax_nyq.set_ylabel("Im{H(iω)}")
    ax_nyq.axhline(0, color="black", linewidth=0.8)
    ax_nyq.axvline(0, color="black", linewidth=0.8)
    ax_nyq.grid(True, linestyle=":")
    ax_nyq.legend()
    ax_nyq.set_aspect("equal", "box")

    fig_nyq.suptitle("Nyquist plots (using scipy.signal.freqresp)", fontsize=14)

    # ---- 保存 ----
    outdir = Path(args.outdir)
    tag = build_tag(args.tau, args.omega0, args.zeta, args.wmin, args.wmax, args.tmax)

    # 例: lti_demo__tau1_om1_z0p3_w0.01-100_t20__step.png
    base = _sanitize(args.prefix)
    stem_step = f"{base}__{tag}__step"
    stem_bode = f"{base}__{tag}__bode"
    stem_nyq  = f"{base}__{tag}__nyquist"

    save_figure(fig_step, outdir, stem_step, args.formats, args.dpi)
    save_figure(fig_bode, outdir, stem_bode, args.formats, args.dpi)
    save_figure(fig_nyq,  outdir, stem_nyq,  args.formats, args.dpi)

    # ---- 表示 or 終了 ----
    if args.no_show:
        plt.close("all")
    else:
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()