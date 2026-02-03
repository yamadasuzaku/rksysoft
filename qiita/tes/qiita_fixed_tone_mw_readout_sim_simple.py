#!/usr/bin/env python3
"""
Fixed-tone microwave readout simulator with:
  (1) Single resonator + complex S21 (I/Q) + vector plot
  (2) Multiple resonators (MW-Mux-like) demonstration

TES / MW-Mux を意識して、S21 は「ピーク型」ではなく
共振でノッチ (減衰) が出る「ノッチ型 (notch-type)」としてモデル化している。

このスクリプトの目的:
- 「S21 がノッチになる共振器」を数式とプロットでイメージする
- fixed-tone readout で I/Q がどう動くかを可視化する
- Q 値や共振周波数などの基礎パラメータを print で確認しながら学ぶ

実行すると、現在のディレクトリに PNG ファイルとして図が保存される:
- single_resonator_linear.png
- single_resonator_db.png
- multi_resonator_time_linear.png
- multi_resonator_static_linear.png
- multi_resonator_static_db.png
"""

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
#  描画用のユーティリティ
# ============================================================

def apply_light_grid(ax):
    """
    学生さん向けに「薄めのグリッド」を統一的に適用する関数。
    grid(True) だけだと線が濃くて見づらいことがあるので、
    alpha, linewidth, linestyle を調整している。
    """
    ax.grid(True, alpha=0.3, linewidth=0.5, linestyle="--")


# ============================================================
#  共通ユーティリティ: 共振器の基礎パラメータを表示する関数
# ============================================================

def resonator_s21_complex(f, f_res, Q):
    """
    単純化された 1ポール共振器の複素伝達関数 S21(f) を返す (ノッチ型)。

    ここでは TES の MW-Mux でよく出てくる
    「通過帯 (|S21|≈1) の中に、共振周波数でノッチが落ちる」
    形を意識して、以下のような簡略モデルを採用する:

        S21(f) = 1 - 1 / (1 + j * x)

      ここで
        x     = (f - f_res) / gamma
        gamma = f_res / (2 Q)  (Half Width at Half Maximum: HWHM)

    ・f が f_res から十分離れているとき:
        x が大きくなり 1 / (1 + j x) ≈ 0 なので
        S21 ≈ 1  (ほぼ全通過)

    ・f = f_res の近傍:
        x = 0 なので 1 / (1 + j * 0) = 1
        よって S21 = 1 - 1 = 0 となり、ノッチ (深い落ち込み) になる。

    実際の MW-Mux では結合の強さや内部損失などでもう少し複雑になるが、
    ここでは「ノッチ型である」という直感をつかむことを目的にしている。
    """
    # ★ f はスカラーでも配列でもよいように計算している
    gamma = f_res / (2.0 * Q)  # HWHM [Hz]
    x = (f - f_res) / gamma
    S21 = 1.0 - 1.0 / (1.0 + 1j * x)
    return S21


def print_resonator_basic_params(f_res, Q, label="Resonator"):
    """
    共振器の基礎パラメータを print で表示する。

    引数:
        f_res : 共振周波数 [Hz]
        Q     : Q値 (loaded Q を想定)
        label : どの共振器か識別するラベル（例: "Res 0"）
    """
    # Half Width at Half Maximum (HWHM)
    gamma = f_res / (2.0 * Q)     # [Hz]
    # Full Width at Half Maximum (FWHM) ~ f_res / Q
    fwhm = f_res / Q              # [Hz]

    # ノッチ共振器の伝達特性を使って、
    # - 共振から十分離れたところ (f_res + 10*gamma)
    # - 共振ちょうど (f_res)
    # での透過率 |S21| を計算してみる
    S21_far = resonator_s21_complex(f_res + 10.0 * gamma, f_res, Q)
    S21_at_res = resonator_s21_complex(f_res, f_res, Q)

    amp_far = np.abs(S21_far)
    amp_res = np.abs(S21_at_res)

    # log10(0) にならないように微小量 1e-15 を足している
    amp_far_db = 20.0 * np.log10(amp_far + 1e-15)
    amp_res_db = 20.0 * np.log10(amp_res + 1e-15)

    print("========================================================")
    print(f"[{label}] 基本パラメータ")
    print(f"  共振周波数 f0 = {f_res:.3e} Hz  ({f_res*1e-9:.3f} GHz)")
    print(f"  Q 値 Q      = {Q:.3e}")
    print(f"  HWHM gamma  = {gamma:.3e} Hz")
    print(f"  FWHM ≈ f0/Q = {fwhm:.3e} Hz  ({fwhm*1e-6:.3f} MHz)")
    print("  共振から十分離れた周波数での透過率 (ほぼ全通過):")
    print(f"    |S21| ≈ {amp_far:.3f}   ({amp_far_db:.2f} dB)")
    print("  共振周波数ちょうどでの透過率 (ノッチの深さ):")
    print(f"    |S21| ≈ {amp_res:.3e}   ({amp_res_db:.2f} dB)")
    print("    -> この簡略モデルでは理想的にノッチが 0 まで落ちる (実機ではここまで深くない)。")
    print("========================================================\n")


# ============================================================
#  (1) 単一共振器: fixed-tone + I/Q
# ============================================================

def simulate_single_resonator(
    f0=5.0e9,
    Q=300.0,           # ★ デフォルト Q を 300 に変更
    df_max=10.0e6,
    tau_ms=10.0,
    t_total_ms=50.0,
    dt_us=20.0,
    t_event_ms=5.0,
    probe_offset_MHz=0.0,
):
    """
    単一共振器の fixed-tone readout を時間領域でシミュレートする。

    - 共振周波数 f_res(t) が、ある時刻 t_event_ms 以降で
      指数関数的にシフトして元に戻る簡単なモデル。
    - 実際には TES への X 線入射に対応しているとイメージすると良い。

    戻り値:
        t_ms      : 時間 [ms]
        f_res_Hz  : 各時刻での共振周波数 [Hz]
        S21_t     : 各時刻で固定トーンに対する S21(t)
        f_probe_Hz: 固定トーンの周波数 [Hz]
    """
    # --- 時間軸の設定 ---
    dt_s = dt_us * 1.0e-6            # サンプリング間隔 [s]
    t_total_s = t_total_ms * 1.0e-3  # 総観測時間 [s]
    t_event_s = t_event_ms * 1.0e-3  # 事象発生時刻 [s]
    tau_s = tau_ms * 1.0e-3          # 緩和時定数 [s]

    # 0 から t_total_s まで dt_s 刻みの時間配列
    # ★ arange は「終点を含まない」ので、少し余裕を持たせている
    t_s = np.arange(0.0, t_total_s, dt_s)
    t_ms = t_s * 1.0e3              # プロットしやすいよう [ms] も作る

    # --- 共振周波数 f_res(t) の時間変化 ---
    f_res_Hz = np.empty_like(t_s)
    for i, t in enumerate(t_s):
        if t < t_event_s:
            # 事象発生前は共振周波数 f0 のまま
            f_res_Hz[i] = f0
        else:
            # 事象発生後は df_max だけシフトして、指数関数的に元に戻る
            f_res_Hz[i] = f0 + df_max * np.exp(-(t - t_event_s) / tau_s)

    # --- fixed-tone の周波数 ---
    # 実際の readout では、共振周波数の近くに一定周波数のトーンを打ちっぱなしにする。
    f_probe_Hz = f0 + probe_offset_MHz * 1.0e6

    # 各時刻での S21(t)（固定トーン f_probe_Hz に対する伝達関数）
    # ★ 引数の順番に注意: (f_probe, f_res(t), Q)
    S21_t = resonator_s21_complex(f_probe_Hz, f_res_Hz, Q)

    return t_ms, f_res_Hz, S21_t, f_probe_Hz


def plot_single_resonator(t_ms, f_res_Hz, S21_t, f_probe_Hz, f0, Q):
    """
    単一共振器の結果をプロットする。

    Figure 1 (リニア表示):
      1) Δf_res(t) [MHz]
      2) |S21(t)| vs Time (linear)
      3) I-Q 平面での軌跡
      4) 静的な共振曲線 |S21(f)| (notch, linear)

    Figure 2 (dB 表示):
      1) S21(t) [dB] vs Time
      2) 静的な共振曲線 S21(f) [dB]

    ★ 各 Figure は PNG ファイルとしても保存する。
    """
    # 共振周波数の変化量 (f_res - f0) を [MHz] 単位で計算
    df_MHz = (f_res_Hz - f0) * 1.0e-6

    # S21 の振幅（透過率）と dB 表示
    amp_t = np.abs(S21_t)                      # |S21| (linear, 透過率)
    amp_db_t = 20.0 * np.log10(amp_t + 1e-15)  # [dB], log(0) 回避のため微小量を足す

    # I/Q 成分を取り出す (I=実部, Q=虚部)
    I_t = np.real(S21_t)
    Q_t = np.imag(S21_t)

    # -------------------------------
    # Figure 1: リニア表示まとめ
    # -------------------------------
    fig1 = plt.figure(figsize=(10, 8))

    # 1) Δf_res(t)
    ax1 = fig1.add_subplot(2, 2, 1)
    ax1.plot(t_ms, df_MHz)
    ax1.set_xlabel("Time [ms]")
    ax1.set_ylabel("Δf_res [MHz]")
    ax1.set_title("Resonance frequency shift vs time")
    apply_light_grid(ax1)

    # 2) |S21(t)| (linear)
    ax2 = fig1.add_subplot(2, 2, 2)
    ax2.plot(t_ms, amp_t)
    ax2.set_xlabel("Time [ms]")
    ax2.set_ylabel("Transmission |S21| (linear)")
    ax2.set_title("Fixed-tone microwave readout (amplitude, linear)")
    apply_light_grid(ax2)

    # 3) I-Q trajectory
    ax3 = fig1.add_subplot(2, 2, 3)
    ax3.plot(I_t, Q_t, "-")
    ax3.set_xlabel("I = Re(S21)")
    ax3.set_ylabel("Q = Im(S21)")
    ax3.set_title("I-Q trajectory at fixed tone")
    apply_light_grid(ax3)
    ax3.set_aspect("equal", "box")

    # 開始点・終了点・最大シフト付近をマーカーで示す
    ax3.plot(I_t[0], Q_t[0], "go", label="start")
    ax3.plot(I_t[-1], Q_t[-1], "ro", label="end")
    idx_event = np.argmax(df_MHz)  # 周波数シフト最大付近
    ax3.plot(I_t[idx_event], Q_t[idx_event], "bo", label="around max shift")
    ax3.legend(loc="best")

    # 4) 静的な共振曲線 |S21(f)| (linear)
    ax4 = fig1.add_subplot(2, 2, 4)
    span_MHz = 30.0
    f_MHz = np.linspace(-span_MHz, span_MHz, 1000)
    f_Hz = f0 + f_MHz * 1.0e6
    S21_static = resonator_s21_complex(f_Hz, f0, Q)
    amp_static = np.abs(S21_static)

    ax4.plot(f_MHz, amp_static, label="|S21(f)| (linear)")
    ax4.axvline((f_probe_Hz - f0) * 1.0e-6, linestyle="--", label="probe tone")
    ax4.set_xlabel("f - f0 [MHz]")
    ax4.set_ylabel("Transmission |S21| (linear)")
    ax4.set_title("Static resonance (notch) and probe frequency (linear)")
    apply_light_grid(ax4)
    ax4.legend(loc="best", fontsize=8)

    plt.tight_layout()

    # ★ 図を保存（カレントディレクトリに PNG として）
    fig1.savefig("single_resonator_linear.png", dpi=150, bbox_inches="tight")
    print("Saved: single_resonator_linear.png")

    plt.show()

    # -------------------------------
    # Figure 2: dB 表示まとめ
    # -------------------------------
    fig2, (ax5, ax6) = plt.subplots(1, 2, figsize=(10, 4))

    # 5) S21(t) [dB] vs time
    ax5.plot(t_ms, amp_db_t)
    ax5.set_xlabel("Time [ms]")
    ax5.set_ylabel("S21 [dB]")
    ax5.set_title("Fixed-tone microwave readout (amplitude, dB)")
    apply_light_grid(ax5)

    # 6) 静的な共振曲線 S21(f) [dB]
    amp_static_db = 20.0 * np.log10(amp_static + 1e-15)
    ax6.plot(f_MHz, amp_static_db)
    ax6.axvline((f_probe_Hz - f0) * 1.0e-6, linestyle="--", label="probe tone")
    ax6.set_xlabel("f - f0 [MHz]")
    ax6.set_ylabel("S21 [dB]")
    ax6.set_title("Static resonance (notch) in dB")
    apply_light_grid(ax6)
    ax6.legend(loc="best", fontsize=8)

    plt.tight_layout()

    # ★ 図を保存
    fig2.savefig("single_resonator_db.png", dpi=150, bbox_inches="tight")
    print("Saved: single_resonator_db.png")

    plt.show()


# ============================================================
#  (2) 複数共振器 (MW-Mux 的なデモ)
# ============================================================

def simulate_multi_resonators(
    n_res=4,
    f0_center=5.0e9,
    spacing_MHz=50.0,
    Q=300.0,           # ★ デフォルト Q を 300 に変更
    df_max=10.0e6,
    tau_ms=10.0,
    t_total_ms=50.0,
    dt_us=20.0,
    t_event_ms=5.0,
    hit_index=1,
):
    """
    n_res 本の共振器を周波数方向に等間隔に並べ、
    そのうち 1 本だけが X 線ヒットで周波数シフトを経験するモデル。

    - f0_center を中心として、spacing_MHz [MHz] 間隔で共振器を並べる。
    - 各共振器ごとにプローブトーンは「その共振器の共振周波数」に固定。
    - hit_index 番目の共振器だけが df_max だけ周波数シフトする。

    戻り値:
        t_ms     : 時間 [ms]
        f0_list  : 各共振器の中心共振周波数 [Hz] の配列 (長さ n_res)
        S21_multi: shape = (n_res, n_t) の複素配列
                   S21_multi[k, i] が「k 番目の共振器の i 番目の時刻での S21」
    """
    # 共振器のインデックス (0, 1, 2, ..., n_res-1) を
    # 中心からの相対オフセット (-..., 0, +...) に変換
    idx_offset = np.arange(n_res) - (n_res - 1) / 2.0
    f0_list = f0_center + idx_offset * spacing_MHz * 1.0e6  # 各共振器の f0

    # --- 時間軸 ---
    dt_s = dt_us * 1.0e-6
    t_total_s = t_total_ms * 1.0e-3
    t_event_s = t_event_ms * 1.0e-3
    tau_s = tau_ms * 1.0e-3

    t_s = np.arange(0.0, t_total_s, dt_s)
    t_ms = t_s * 1.0e3
    n_t = len(t_s)

    # S21_multi[k, i] : k 番目の共振器の i 番時刻での S21
    S21_multi = np.zeros((n_res, n_t), dtype=complex)

    # 各共振器ごとに、時間変化する f_res_k(t) と S21_k(t) を計算
    for k in range(n_res):
        f0_k = f0_list[k]
        f_probe_k = f0_k  # ここでは probe トーンを共振周波数に固定（オフセット 0）

        f_res_k = np.empty_like(t_s)
        for i, t in enumerate(t_s):
            if k == hit_index and t >= t_event_s:
                # この共振器だけ事象によって df_max シフトする
                f_res_k[i] = f0_k + df_max * np.exp(-(t - t_event_s) / tau_s)
            else:
                # 他の共振器、または事象前は f0_k のまま
                f_res_k[i] = f0_k

        # ★ ここも引数の順序: (f_probe_k, f_res_k(t), Q)
        S21_k = resonator_s21_complex(f_probe_k, f_res_k, Q)
        S21_multi[k, :] = S21_k

    return t_ms, f0_list, S21_multi


def plot_multi_resonators(t_ms, f0_center, f0_list, S21_multi, Q, spacing_MHz=50.0):
    """
    複数共振器の |S21_k(t)| と静的共振曲線をプロット。

    ・Figure 1: time-domain |S21_k(t)| (linear)
    ・Figure 2: static |S21(f)| (linear)
    ・Figure 3: static S21(f) [dB]

    ★ 各 Figure は PNG ファイルとしても保存する。
    """
    n_res, n_t = S21_multi.shape

    # -------------------------------
    # Figure 1: time-domain amplitude (linear)
    # -------------------------------
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    for k in range(n_res):
        amp_k = np.abs(S21_multi[k, :])
        label = f"Resonator {k} (f0 = {f0_list[k]*1e-9:.3f} GHz)"
        ax1.plot(t_ms, amp_k, label=label)
    ax1.set_xlabel("Time [ms]")
    ax1.set_ylabel("|S21| at each probe tone (linear)")
    ax1.set_title("MW-Mux-like: amplitude at each tone vs time (notch-type, linear)")
    apply_light_grid(ax1)
    ax1.legend(loc="best", fontsize=8)

    plt.tight_layout()
    fig1.savefig("multi_resonator_time_linear.png", dpi=150, bbox_inches="tight")
    print("Saved: multi_resonator_time_linear.png")
    plt.show()

    # -------------------------------
    # Figure 2: static resonance curves (linear)
    # -------------------------------
    fig2, ax2 = plt.subplots(figsize=(7, 4))

    span_total_MHz = spacing_MHz * (n_res + 1)
    f_MHz = np.linspace(-span_total_MHz, span_total_MHz, 2000)
    f_Hz = f0_center + f_MHz * 1.0e6

    for k in range(n_res):
        f0_k = f0_list[k]
        S_static_k = resonator_s21_complex(f_Hz, f0_k, Q)
        amp_k = np.abs(S_static_k)
        label = f"Res {k} (f0={f0_k*1e-9:.3f} GHz)"
        ax2.plot(f_MHz, amp_k, label=label)

        # 各共振器の位置に縦線を引いて分かりやすくする
        offset_k_MHz = (f0_k - f0_center) * 1.0e-6
        ax2.axvline(offset_k_MHz, linestyle="--", alpha=0.3)

    ax2.set_xlabel("f - f_center [MHz]")
    ax2.set_ylabel("Transmission |S21| (linear)")
    ax2.set_title("Static resonance curves (notch-type, linear)")
    apply_light_grid(ax2)
    ax2.legend(loc="best", fontsize=8)

    plt.tight_layout()
    fig2.savefig("multi_resonator_static_linear.png", dpi=150, bbox_inches="tight")
    print("Saved: multi_resonator_static_linear.png")
    plt.show()

    # -------------------------------
    # Figure 3: static resonance curves (dB)
    # -------------------------------
    fig3, ax3 = plt.subplots(figsize=(7, 4))

    for k in range(n_res):
        f0_k = f0_list[k]
        S_static_k = resonator_s21_complex(f_Hz, f0_k, Q)
        amp_k = np.abs(S_static_k)
        amp_k_db = 20.0 * np.log10(amp_k + 1e-15)
        label = f"Res {k} (f0={f0_k*1e-9:.3f} GHz)"
        ax3.plot(f_MHz, amp_k_db, label=label)

        offset_k_MHz = (f0_k - f0_center) * 1.0e-6
        ax3.axvline(offset_k_MHz, linestyle="--", alpha=0.3)

    ax3.set_xlabel("f - f_center [MHz]")
    ax3.set_ylabel("S21 [dB]")
    ax3.set_title("Static resonance curves (notch-type, dB)")
    apply_light_grid(ax3)
    ax3.legend(loc="best", fontsize=8)

    plt.tight_layout()
    fig3.savefig("multi_resonator_static_db.png", dpi=150, bbox_inches="tight")
    print("Saved: multi_resonator_static_db.png")
    plt.show()


def plot_single_resonator_time_series(t_ms, S21_t, f_res_Hz=None, f0=None):
    """
    時間を横軸にして、
      - I(t) = Re(S21(t))
      - Q(t) = Im(S21(t))
      - |S21(t)| (linear)
    を 3 段のサブプロットで表示する。

    第3パネルでは、
      - 左軸 : |S21(t)| (linear)
      - 右軸 : Δf_res(t) [MHz]（オプション）
    とし、色と凡例で混乱しないようにする。
    """
    # --- I, Q, 振幅 ---
    I_t = np.real(S21_t)
    Q_t = np.imag(S21_t)
    amp_t = np.abs(S21_t)

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    # ===============================
    # 1) I(t)
    # ===============================
    ax_I = axes[0]
    ax_I.plot(t_ms, I_t, color="tab:blue")
    ax_I.set_ylabel("I(t) = Re(S21)")
    ax_I.set_title("Time-domain view: I(t), Q(t), |S21(t)|")
    apply_light_grid(ax_I)

    # ===============================
    # 2) Q(t)
    # ===============================
    ax_Q = axes[1]
    ax_Q.plot(t_ms, Q_t, color="tab:green")
    ax_Q.set_ylabel("Q(t) = Im(S21)")
    apply_light_grid(ax_Q)

    # ===============================
    # 3) |S21(t)| + Δf(t)
    # ===============================
    ax_A = axes[2]

    # 左軸: |S21|
    line_amp, = ax_A.plot(
        t_ms, amp_t,
        color="tab:blue",
        label="|S21(t)| (linear)"
    )
    ax_A.set_ylabel("|S21(t)| (linear)", color="tab:blue")
    ax_A.tick_params(axis="y", labelcolor="tab:blue")
    ax_A.set_xlabel("Time [ms]")
    apply_light_grid(ax_A)

    lines = [line_amp]
    labels = [line_amp.get_label()]

    # 右軸: Δf_res(t)
    if f_res_Hz is not None and f0 is not None:
        df_MHz = (f_res_Hz - f0) * 1.0e-6

        ax_df = ax_A.twinx()
        line_df, = ax_df.plot(
            t_ms, df_MHz,
            linestyle="--",
            color="tab:orange",
            label="Δf_res(t) [MHz]"
        )
        ax_df.set_ylabel("Δf_res(t) [MHz]", color="tab:orange")
        ax_df.tick_params(axis="y", labelcolor="tab:orange")

        # 凡例用に追加
        lines.append(line_df)
        labels.append(line_df.get_label())

    # 凡例（左右軸をまたぐので手動指定）
    ax_A.legend(lines, labels, loc="best", fontsize=9)

    plt.tight_layout()

    # 保存
    fig.savefig(
        "single_resonator_time_I_Q_amp_df.png",
        dpi=150,
        bbox_inches="tight"
    )
    print("Saved: single_resonator_time_I_Q_amp_df.png")

    plt.show()


# ============================================================
#  main: デモ用ラッパ関数
# ============================================================

def demo_single_resonator():
    """単一共振器 + I/Q デモ (ノッチ型 S21)"""
    # ここで設定しているのは「5 GHz, Q=300 の共振器」をイメージ
    f0 = 5.0e9       # 共振周波数 [Hz]
    Q = 300.0        # ★ Q 値 (比較的低めにして共振を太く見せる)
    df_max = 10.0e6  # 事象による最大周波数シフト [Hz]
    tau_ms = 10.0    # 緩和時定数 [ms]
    t_total_ms = 50.0
    dt_us = 20.0
    t_event_ms = 5.0
    probe_offset_MHz = 0.0  # probe トーンを f0 ど真ん中に置く

    # --- 共振器の基本パラメータを print しておく ---
    print_resonator_basic_params(f0, Q, label="Single resonator")

    # シミュレーション実行
    t_ms, f_res_Hz, S21_t, f_probe_Hz = simulate_single_resonator(
        f0=f0,
        Q=Q,
        df_max=df_max,
        tau_ms=tau_ms,
        t_total_ms=t_total_ms,
        dt_us=dt_us,
        t_event_ms=t_event_ms,
        probe_offset_MHz=probe_offset_MHz,
    )

    # プロット
    plot_single_resonator(t_ms, f_res_Hz, S21_t, f_probe_Hz, f0, Q)

    # ★ 追加: 時間 vs I, Q, |S21| の図
    plot_single_resonator_time_series(t_ms, S21_t, f_res_Hz=f_res_Hz, f0=f0)


def demo_multi_resonators():
    """複数共振器 (MW-Mux 的, ノッチ型 S21) デモ"""
    n_res = 4
    f0_center = 5.0e9
    spacing_MHz = 50.0
    Q = 300.0       # ★ こちらも Q=300
    df_max = 10.0e6
    tau_ms = 10.0
    t_total_ms = 50.0
    dt_us = 20.0
    t_event_ms = 5.0
    hit_index = 1  # このインデックスの共振器だけ周波数シフトさせる

    # シミュレーション実行
    t_ms, f0_list, S21_multi = simulate_multi_resonators(
        n_res=n_res,
        f0_center=f0_center,
        spacing_MHz=spacing_MHz,
        Q=Q,
        df_max=df_max,
        tau_ms=tau_ms,
        t_total_ms=t_total_ms,
        dt_us=dt_us,
        t_event_ms=t_event_ms,
        hit_index=hit_index,
    )

    # --- 各共振器の基本パラメータを print ---
    print("=== Multi-resonator configuration ===")
    for k, f0_k in enumerate(f0_list):
        print_resonator_basic_params(f0_k, Q, label=f"Resonator {k}")

    # プロット
    plot_multi_resonators(
        t_ms=t_ms,
        f0_center=f0_center,
        f0_list=f0_list,
        S21_multi=S21_multi,
        Q=Q,
        spacing_MHz=spacing_MHz,
    )


def main():
    # ★ 学生向けには main() の中でどんな処理が呼ばれているかも重要
    #   - 単一共振器のデモ
    #   - 複数共振器 (MW-Mux 的) のデモ
    demo_single_resonator()
    demo_multi_resonators()


if __name__ == "__main__":
    main()
