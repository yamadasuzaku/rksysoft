#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
丁寧コメント版

このスクリプトは：

1) calc_branchingratios(rate) で「真のレート lambda に対して，各グレードが出る確率 p_g(lambda)」を作る
2) Hp/Mp/Ms（fit_grades）だけを使って，Poisson 尤度から lambda を推定（MLE）
3) 尤度比（profile likelihood）で信頼区間（CI）を求める
4) シミュレーション（Poisson でカウント生成）を繰り返し，推定の安定性や Lp/Ls 予測の不確かさをまとめて可視化する

用語メモ：
- lambda: 1 pixel あたりの真のレート [counts/s/pixel]
- T: 露光時間 [s]
- p_g(lambda): グレード g が出る確率（分岐比）
- K_g: 観測（シミュレーション）されたグレード g のカウント
- mu_g = T * lambda * p_g(lambda): Poisson の期待値
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, brentq

# ============================================================
# 分岐比（branching ratios）を計算する関数
# ============================================================
def calc_branchingratios(rate, debug=False):
    """
    分岐比を計算する関数。
    武田さんSPIEに準じるが検証が必要な式。

    Parameters
    ----------
    rate : float or array-like
        真のレート（lambda）[counts/s/pixel]。スカラーでも配列でもOK。
    debug : bool
        True にすると内部の checksum を表示する。

    Returns
    -------
    Hp_bl_cor, Mp_bl, Ms_bl, Lp_bl_cor, Ls_bl_cor : float or ndarray
        それぞれのグレードの確率（分岐比）。
        入力が配列なら配列で返る（同じ形状）。
    """
    # 12.5 kHz のクロック（= 80 micro-second 周期）
    clock = 12500  # 12.5kHz = 80us

    # ここは装置（または処理）に固有の「時間窓」っぽい係数
    # dtHR, dtMR, dtLR は rate に掛けて exp(-dt*rate) の形で使う
    dtHR = (1.0 / clock) * 874
    dtMR = (1.0 / clock) * 219
    dtLR = (1.0 / clock) * 15

    # e^{-dt * rate} をそれぞれ計算
    exp_hr = np.exp(-1.0 * dtHR * rate)
    exp_mr = np.exp(-1.0 * dtMR * rate)
    exp_lr = np.exp(-1.0 * dtLR * rate)

    # --- 分岐比の「生（未正規化）」の計算 ---
    # ここは式が本体なので、内容は変えない（そのまま）
    Hp_bl_cor = exp_hr * (exp_hr + 1 - exp_lr)
    Mp_bl = exp_hr * (exp_mr - exp_hr)
    Ms_bl = exp_mr * (exp_mr - exp_hr)
    Lp_bl_cor = exp_hr * (exp_lr - exp_mr)
    Ls_bl_cor = (1 - exp_mr) * (1 + exp_mr - exp_hr) - exp_hr * (1 - exp_lr)

    # 5つの確率の和（= 正規化係数）を計算
    checksum = Hp_bl_cor + Mp_bl + Ms_bl + Lp_bl_cor + Ls_bl_cor

    # --- 正規化：合計が 1 になるように割る ---
    Hp_bl_cor /= checksum
    Mp_bl /= checksum
    Ms_bl /= checksum
    Lp_bl_cor /= checksum
    Ls_bl_cor /= checksum

    # 念のため正規化後の合計も計算（debug用）
    checksum_confirm = Hp_bl_cor + Mp_bl + Ms_bl + Lp_bl_cor + Ls_bl_cor

    if debug:
        print("########## debug in calc_branchingratios ##########", debug)
        print("checksum", checksum)
        print("checksum_confirm", checksum_confirm)

    return Hp_bl_cor, Mp_bl, Ms_bl, Lp_bl_cor, Ls_bl_cor


# ============================================================
# 1) p_of(g, lam) を作る（calc_branchingratios の結果を補間して返す）
# ============================================================
def build_branching_p_of(rate_max: float, rate_grid_n: int, itypenames):
    """
    calc_branchingratios(rate_grid) をあらかじめ計算しておき、
    任意の lambda に対して確率 p_g(lambda) を返す関数 p_of(g, lam) を作る。

    ポイント：
    - rate_grid を用意して calc_branchingratios をベクトル計算（高速）
    - 途中の lambda は np.interp で線形補間して近似
    """
    # 0 から rate_max までを等間隔でサンプリング
    rate_grid = np.linspace(0.0, float(rate_max), int(rate_grid_n))

    # rate_grid の全点に対して分岐比をまとめて計算（配列が返る）
    br = calc_branchingratios(rate_grid)

    def _get_br_array(g):
        """
        グレード g に対応する分岐比配列（rate_grid 上の値）を取り出す。

        br が dict 的にアクセスできる場合（br[g]）も想定して try しているが、
        普通はタプル（Hp, Mp, Ms, Lp, Ls）の順なので、itypenames から index を引く。
        """
        # dict / structured: br[g] が可能ならそれを使う
        try:
            return br[g]
        except Exception:
            pass

        # list/ndarray: itypenames の順番に従って取り出す
        idx = list(itypenames).index(g)
        return br[idx]

    def p_of(g, lam):
        """
        p_of(g, lam) = 「lambda のときグレード g が出る確率」を返す。

        - lam が不正（<=0, inf, nan）なら 0
        - br がスカラーならそのまま返す
        - br が配列なら、rate_grid 上の値を補間して返す
        """
        lam = float(lam)

        # 物理的に lambda<=0 は意味がないので 0 を返す
        if lam <= 0.0 or not np.isfinite(lam):
            return 0.0

        arr = _get_br_array(g)

        # --- もし arr がスカラーなら（通常はここに来ないが安全のため） ---
        if np.ndim(arr) == 0:
            v = float(arr)
            return v if (np.isfinite(v) and v > 0.0) else 0.0

        # --- arr が rate_grid に沿った配列なら補間する ---
        # left/right に np.nan を入れて範囲外を弾きやすくしている
        v = float(np.interp(lam, rate_grid, np.asarray(arr), left=np.nan, right=np.nan))

        # 補間結果が不正なら 0
        if not np.isfinite(v) or v <= 0.0:
            return 0.0
        return v

    return p_of


# ============================================================
# 2) fit_grades のみを使った Poisson 対数尤度 logL(lambda)
# ============================================================
def logL_lambda(lam, T, K_by_grade, p_of, fit_grades):
    """
    Poisson の対数尤度 log L(lambda) を計算する。

    観測（あるいはシミュレーション）で得たカウント K_g に対し、
    期待値 mu_g = T * lambda * p_g(lambda)
    として

        log L = Σ_g [ K_g * log(mu_g) - mu_g ]   （定数 -log(K_g!) は省略）

    を計算する。

    注意：
    - K_g! は lambda に依らないので、最適化（argmax）には不要。
    """
    lam = float(lam)
    if lam <= 0.0 or not np.isfinite(lam):
        return -np.inf

    ll = 0.0
    for g in fit_grades:
        # 辞書から K_g を取り出す（無ければ 0）
        Kg = int(K_by_grade.get(g, 0))

        # p_g(lambda)
        pg = float(p_of(g, lam))

        # Poisson の期待値 mu_g
        mu = T * lam * pg

        # mu が不正ならこの lambda はダメ（尤度ゼロ扱い）
        if mu <= 0.0 or not np.isfinite(mu):
            return -np.inf

        # 対数尤度の加算（-log(K!) は省略）
        ll += Kg * np.log(mu) - mu

    return float(ll)


# ------------------------------------------------------------
# CI（信頼区間）を profile likelihood で解く
# 2*(ll_hat - ll(lam)) = ci_delta_2logL を満たす lam を探す
# ------------------------------------------------------------
def mle_and_profile_ci(
    T,
    K_by_grade,
    p_of,
    fit_grades,
    lam_min,
    lam_max,
    ci_delta_2logL=1.0,
    scan_points=1500,  # brentq が失敗したときのフォールバック用
):
    """
    MLE（最尤推定値 lam_hat）と profile likelihood による CI を返す。

    手順：
    1) minimize_scalar(method="bounded") で -logL を最小化（= logL を最大化）
    2) f(lam) = 2*(ll_hat - ll(lam)) - delta の根を brentq で両側（lo/hi）から探す
    3) brentq がダメなら logspace グリッドでスキャンして CI を近似する

    Returns
    -------
    lam_hat : float
        最尤推定値（MLE）
    (lam_lo, lam_hi) : tuple(float, float)
        信頼区間の下限・上限（見つからなければ NaN）
    """

    # 最小化したい関数：nll = -logL
    def nll(x):
        return -logL_lambda(x, T, K_by_grade, p_of, fit_grades)

    # ---- 1) MLE（bounded） ----
    res = minimize_scalar(nll, bounds=(lam_min, lam_max), method="bounded")
    lam_hat = float(res.x) if res.success else np.nan
    ll_hat = logL_lambda(lam_hat, T, K_by_grade, p_of, fit_grades) if np.isfinite(lam_hat) else -np.inf

    if not np.isfinite(lam_hat) or not np.isfinite(ll_hat):
        return np.nan, (np.nan, np.nan)

    # ---- 2) CI 境界を定義する関数 f(lam) ----
    # f(lam) = 0 となる lam が CI 境界
    delta = float(ci_delta_2logL)

    def f(lam):
        ll = logL_lambda(lam, T, K_by_grade, p_of, fit_grades)
        if not np.isfinite(ll):
            return np.nan
        return 2.0 * (ll_hat - ll) - delta

    def solve_one_side(side: str):
        """
        CI の片側（lo または hi）を解く。

        side: "lo" or "hi"
        戻り値：境界 lambda（失敗時は np.nan）
        """
        eps = 1e-12
        lo_bound = max(float(lam_min), eps)
        hi_bound = float(lam_max)

        # MLE が境界に張り付いていると、その側の CI は定義しにくい
        if lam_hat <= lo_bound * (1.0 + 1e-12) and side == "lo":
            return lo_bound
        if lam_hat >= hi_bound * (1.0 - 1e-12) and side == "hi":
            return hi_bound

        # --- lo 側：lam_hat から小さい方に向かって bracket を作る ---
        if side == "lo":
            # 根を含む区間 [x1, x2] を探したい
            # lam_hat では f(lam_hat) ≈ -delta (<0)
            # lam を離すと f が増えて 0 を超える点が出るはず、という前提
            x2 = max(lo_bound, lam_hat)
            x1 = x2

            for _ in range(80):
                # 1.5 倍ずつ小さくしていく（対数的探索）
                x1 = max(lo_bound, x1 / 1.5)

                v1 = f(x1)
                v2 = f(x2)

                # brentq には「符号が異なる」区間が必要（v1>0, v2<0）
                if np.isfinite(v1) and np.isfinite(v2) and (v1 > 0.0) and (v2 < 0.0):
                    try:
                        return float(brentq(lambda z: f(z), x1, x2, maxiter=200))
                    except Exception:
                        break

                if x1 <= lo_bound * (1.0 + 1e-12):
                    break

            return np.nan

        # --- hi 側：lam_hat から大きい方に向かって bracket を作る ---
        else:  # "hi"
            x1 = min(hi_bound, lam_hat)
            x2 = x1

            for _ in range(80):
                # 1.5 倍ずつ大きくしていく（対数的探索）
                x2 = min(hi_bound, x2 * 1.5)

                v1 = f(x1)
                v2 = f(x2)

                # hi 側は v1<0, v2>0 の区間を探す
                if np.isfinite(v1) and np.isfinite(v2) and (v1 < 0.0) and (v2 > 0.0):
                    try:
                        return float(brentq(lambda z: f(z), x1, x2, maxiter=200))
                    except Exception:
                        break

                if x2 >= hi_bound * (1.0 - 1e-12):
                    break

            return np.nan

    lam_lo = solve_one_side("lo")
    lam_hi = solve_one_side("hi")

    # ---- 3) brentq が失敗したらフォールバック（グリッド探索） ----
    if not (np.isfinite(lam_lo) and np.isfinite(lam_hi) and lam_lo > 0 and lam_hi > 0 and lam_lo <= lam_hat <= lam_hi):
        lo = max(float(lam_min), 1e-12)
        hi = max(float(lam_max), lo * 1.001)

        # 対数空間でスキャン（桁の広い探索に強い）
        grid = np.logspace(np.log10(lo), np.log10(hi), int(scan_points))

        # grid 上で ll を評価
        ll_grid = np.array([logL_lambda(x, T, K_by_grade, p_of, fit_grades) for x in grid], dtype=float)

        # 尤度比の 2ΔlogL
        d2 = 2.0 * (ll_hat - ll_grid)

        # d2 <= delta が CI に入る条件
        ok = np.isfinite(d2) & (d2 <= delta)

        if np.any(ok):
            idx = np.where(ok)[0]
            lam_lo = float(grid[idx[0]])
            lam_hi = float(grid[idx[-1]])
        else:
            lam_lo, lam_hi = np.nan, np.nan

    return lam_hat, (lam_lo, lam_hi)


# ============================================================
# 3) シミュレーション：K_g ~ Poisson(T * lam_true * p_g(lam_true))
# ============================================================
def simulate_counts(T, lam_true, p_of, itypenames, rng):
    """
    各グレード g について期待値 mu_g を作り、Poisson 乱数でカウント K_g を生成する。

    K を辞書で返す（key がグレード、value が整数カウント）。
    """
    K = {}
    for g in itypenames:
        mu = T * float(lam_true) * float(p_of(g, lam_true))
        mu = max(mu, 0.0)
        K[g] = int(rng.poisson(mu))
    return K


# ============================================================
# 4) Sweep & summarize
#    (NEW) target の予測カウントに対して band（中央値など）も保存
# ============================================================
def run_sweep(lam_grid, T, nrep, seed, p_of, itypenames,
              fit_grades, target_grades, rate_max, scan_points=1500):
    """
    lam_true（明るさ）を複数点（lam_grid）で掃引して、
    各点で nrep 回の Poisson シミュレーションを行い、

    - Hp/Mp/Ms（fit_grades）だけで lambda を推定（lam_hat）
    - CI（lam_lo, lam_hi）も求める
    - さらに Lp/Ls（target_grades）の予測期待値 M_g について
      Mhat / Mlo / Mhi を保存し、band の統計をまとめる

    を行う。
    """
    rng = np.random.default_rng(seed)
    out = []

    for lam_true in lam_grid:
        lam_true = float(lam_true)

        # 推定された lambda の集計用
        lam_hats = []
        lam_ciw_rel = []

        # target グレードの「相対CI幅」をためる（従来の診断量）
        target_ciw_rel = {g: [] for g in target_grades}

        # (NEW) target グレードの絶対カウント（期待値）の band 用
        target_Mhat = {g: [] for g in target_grades}
        target_Mlo  = {g: [] for g in target_grades}
        target_Mhi  = {g: [] for g in target_grades}

        ok_count = 0

        for _ in range(int(nrep)):
            # --- 観測（擬似）データ生成 ---
            K = simulate_counts(T, lam_true, p_of, itypenames, rng)

            # fit に使うグレードで「カウントがゼロ」だと尤度が死ぬのでスキップ
            K_fit_sum = sum(int(K.get(g, 0)) for g in fit_grades)
            if K_fit_sum <= 0:
                continue

            # --- MLE と CI 推定 ---
            lam_hat, (lam_lo, lam_hi) = mle_and_profile_ci(
                T, K, p_of, fit_grades,
                lam_min=1e-8, lam_max=float(rate_max),
                ci_delta_2logL=1.0, scan_points=int(scan_points)
            )
            if not np.isfinite(lam_hat):
                continue

            ok_count += 1
            lam_hats.append(lam_hat)

            # lambda の相対 CI 幅（(hi-lo)/hat）を保存（診断用）
            if np.isfinite(lam_lo) and np.isfinite(lam_hi) and lam_hat > 0:
                lam_ciw_rel.append((lam_hi - lam_lo) / lam_hat)
            else:
                lam_ciw_rel.append(np.nan)

            # --- target グレード（例：Lp/Ls）の予測 ---
            for g in target_grades:
                # 推定値 lam_hat に基づく期待カウント
                Mhat = T * lam_hat * p_of(g, lam_hat)

                # 推定CI（lam_lo, lam_hi）をそのまま伝播させた期待カウントの範囲
                if np.isfinite(lam_lo) and np.isfinite(lam_hi):
                    Mlo = T * lam_lo * p_of(g, lam_lo)
                    Mhi = T * lam_hi * p_of(g, lam_hi)
                else:
                    Mlo, Mhi = np.nan, np.nan

                target_Mhat[g].append(Mhat)
                target_Mlo[g].append(Mlo)
                target_Mhi[g].append(Mhi)

                # 相対 CI 幅（(Mhi-Mlo)/Mhat）を保存（診断用）
                if np.isfinite(Mlo) and np.isfinite(Mhi) and np.isfinite(Mhat) and Mhat > 0:
                    target_ciw_rel[g].append((Mhi - Mlo) / Mhat)
                else:
                    target_ciw_rel[g].append(np.nan)

        # --- lam_true ごとに統計量をまとめる ---
        ok_frac = ok_count / float(nrep)
        lam_hats = np.array(lam_hats, float)
        lam_ciw_rel = np.array(lam_ciw_rel, float)

        rec = {
            "lam_true": lam_true,
            "ok_frac": ok_frac,
            "lam_hat_med": np.nanmedian(lam_hats) if ok_count > 0 else np.nan,
            "lam_hat_p16": np.nanpercentile(lam_hats, 16) if ok_count > 0 else np.nan,
            "lam_hat_p84": np.nanpercentile(lam_hats, 84) if ok_count > 0 else np.nan,
            "lam_ciw_rel_med": np.nanmedian(lam_ciw_rel) if ok_count > 0 else np.nan,
        }

        for g in target_grades:
            # target の相対 CI 幅（中央値）
            arr = np.array(target_ciw_rel[g], float)
            rec[f"target_{g}_ciw_rel_med"] = np.nanmedian(arr) if ok_count > 0 else np.nan

            # (NEW) absolute band summaries
            Mh = np.array(target_Mhat[g], float)
            Ml = np.array(target_Mlo[g], float)
            Mu = np.array(target_Mhi[g], float)

            rec[f"target_{g}_Mhat_med"] = np.nanmedian(Mh) if ok_count > 0 else np.nan
            rec[f"target_{g}_Mhat_p16"] = np.nanpercentile(Mh, 16) if ok_count > 0 else np.nan
            rec[f"target_{g}_Mhat_p84"] = np.nanpercentile(Mh, 84) if ok_count > 0 else np.nan

            # Mlo/Mhi 自体の分布の中央値で band を作る（頑健なまとめ方）
            rec[f"target_{g}_Mlo_med"] = np.nanmedian(Ml) if ok_count > 0 else np.nan
            rec[f"target_{g}_Mhi_med"] = np.nanmedian(Mu) if ok_count > 0 else np.nan

        out.append(rec)

    return out


# ============================================================
# 5) 理論曲線：mu_g(lambda) を描く（期待カウントの見通し確認）
# ============================================================
def plot_expected_counts(lam_grid, T, p_of, itypenames, fit_grades, target_grades, title_suffix=""):
    """
    「このくらいの明るさなら、各グレードの期待カウントはいくつ？」を
    事前に確認するためのプロット群。

    - (1) 各グレードごとの期待カウント mu_g
    - (2) fit_grades の合計期待カウント
    - (3) 確率の和 sum_g p_g(lambda)（~1 になっているか）
    """
    lam_grid = np.asarray(lam_grid, float)

    # 各グレード g について mu_g = T * lambda * p_g(lambda)
    mu = {g: T * lam_grid * np.array([p_of(g, x) for x in lam_grid], dtype=float) for g in itypenames}

    # (1) per-grade mu_g
    plt.figure()
    plt.xscale("log")
    plt.yscale("log")
    for g in itypenames:
        plt.plot(lam_grid, np.clip(mu[g], 1e-30, None), marker="o", label=f"mu[{g}]")
    plt.xlabel("lambda_true [c/s/pixel]")
    plt.ylabel("expected counts mu_g = T * lambda * p_g(lambda)")
    plt.grid(alpha=0.2)
    plt.title(f"Expected counts per grade {title_suffix}".strip())
    plt.legend(loc="best", fontsize=9)
    plt.show()

    # (2) fit-grades total expected counts
    mu_fit = np.zeros_like(lam_grid)
    for g in fit_grades:
        mu_fit += mu[g]

    plt.figure()
    plt.xscale("log")
    plt.yscale("log")
    plt.plot(lam_grid, np.clip(mu_fit, 1e-30, None), marker="o")
    plt.xlabel("lambda_true [c/s/pixel]")
    plt.ylabel("sum_{g in fit} mu_g  (expected fit-grade counts)")
    plt.grid(alpha=0.2)
    plt.title(f"Expected counts used for fitting (fit_grades) {title_suffix}".strip())
    plt.show()

    # (3) sum_p(lam) diagnostic
    sum_p = np.zeros_like(lam_grid)
    for g in itypenames:
        sum_p += np.array([p_of(g, x) for x in lam_grid], dtype=float)

    plt.figure()
    plt.xscale("log")
    plt.plot(lam_grid, sum_p, marker="o")
    plt.xlabel("lambda_true [c/s/pixel]")
    plt.ylabel("sum_g p_g(lambda)  (should be ~1 if p is a probability)")
    plt.grid(alpha=0.2)
    plt.title(f"Branching normalization check {title_suffix}".strip())
    plt.show()


# ============================================================
# 6) 結果のサマリープロット（NEW: band plot）
# ============================================================
def plot_summary(summary, target_grades):
    """
    run_sweep の出力 summary（辞書のリスト）から、
    見通しが良いように図をいくつか描く。
    """
    lam_true = np.array([r["lam_true"] for r in summary], float)
    ok_frac  = np.array([r["ok_frac"] for r in summary], float)

    lam_hat = np.array([r["lam_hat_med"] for r in summary], float)
    lam_lo  = np.array([r["lam_hat_p16"] for r in summary], float)
    lam_hi  = np.array([r["lam_hat_p84"] for r in summary], float)

    # --- (A) 推定が成功した割合（fit_grades にカウントがある割合）
    plt.figure()
    plt.xscale("log")
    plt.plot(lam_true, ok_frac, marker="o")
    plt.ylim(-0.05, 1.05)
    plt.xlabel("lambda_true [c/s/pixel]")
    plt.ylabel("success fraction (fit_grades have counts)")
    plt.grid(alpha=0.2)
    plt.title("Estimator success vs brightness")
    plt.show()

    # --- (B) lambda の band（中央値 + p16/p84）
    plt.figure()
    plt.xscale("log")
    plt.yscale("log")

    good = np.isfinite(lam_lo) & np.isfinite(lam_hi) & (lam_lo > 0) & (lam_hi > 0)
    plt.fill_between(lam_true[good], lam_lo[good], lam_hi[good], alpha=0.25, label="lambda: p16–p84 band")

    plt.plot(lam_true, lam_hat, marker="o", label="lambda_hat (median)")
    plt.plot(lam_true, lam_true, linestyle="--", alpha=0.6, label="y=x (truth)")

    plt.xlabel("lambda_true [c/s/pixel]")
    plt.ylabel("estimated lambda")
    plt.grid(alpha=0.2)
    plt.title("Lambda inference (median with 68% band)")
    plt.legend(loc="best")
    plt.show()

    # --- (C) lambda の相対CI幅（診断用）
    lam_ciw = np.array([r["lam_ciw_rel_med"] for r in summary], float)
    plt.figure()
    plt.xscale("log")
    plt.yscale("log")
    plt.plot(lam_true, lam_ciw, marker="o")
    plt.xlabel("lambda_true [c/s/pixel]")
    plt.ylabel("median relative CI width of lambda")
    plt.grid(alpha=0.2)
    plt.title("Lambda relative CI width (diagnostic)")
    plt.show()

    # --- (D) target グレード（Lp/Ls）の期待カウント band
    for g in target_grades:
        Mhat = np.array([r[f"target_{g}_Mhat_med"] for r in summary], float)
        Mlo  = np.array([r[f"target_{g}_Mlo_med"] for r in summary], float)
        Mhi  = np.array([r[f"target_{g}_Mhi_med"] for r in summary], float)

        plt.figure()
        plt.xscale("log")
        plt.yscale("log")

        goodM = np.isfinite(Mlo) & np.isfinite(Mhi) & (Mlo > 0) & (Mhi > 0)
        plt.fill_between(lam_true[goodM], Mlo[goodM], Mhi[goodM], alpha=0.25,
                         label=f"grade {g}: CI band (med of Mlo/Mhi)")
        plt.plot(lam_true, Mhat, marker="o", label=f"grade {g}: Mhat (median)")

        plt.xlabel("lambda_true [c/s/pixel]")
        plt.ylabel(f"predicted expected counts M_g = T*lambda*p_g(lambda)  (grade={g})")
        plt.grid(alpha=0.2)
        plt.title(f"Target grade {g}: predicted expected counts band")
        plt.legend(loc="best")
        plt.show()

        # Relative-width diagnostic (as before)
        ciw = np.array([r[f"target_{g}_ciw_rel_med"] for r in summary], float)
        plt.figure()
        plt.xscale("log")
        plt.yscale("log")
        plt.plot(lam_true, ciw, marker="o")
        plt.xlabel("lambda_true [c/s/pixel]")
        plt.ylabel(f"median relative CI width of predicted counts (grade={g})")
        plt.grid(alpha=0.2)
        plt.title(f"Target grade {g}: relative uncertainty (diagnostic)")
        plt.show()


# ============================================================
# 7) main（コマンドライン引数の受け取り〜実行）
# ============================================================
def main():
    # argparse：コマンドラインからパラメータを受け取る（実験条件を変えて実行しやすい）
    ap = argparse.ArgumentParser(
        description="Quick simulation check for profile-lik Lp/Ls inference using calc_branchingratios()."
    )

    # 露光時間（観測時間）
    ap.add_argument("--T", type=float, default=2000.0, help="Exposure time [s]")

    # lam_true（真の明るさ）を掃引する範囲
    ap.add_argument("--lam_min", type=float, default=1e-2, help="Min lambda_true [c/s/pix]")
    ap.add_argument("--lam_max", type=float, default=40.0, help="Max lambda_true [c/s/pix]")

    # lam_true の点数（logspace で並べる）
    ap.add_argument("--nlam", type=int, default=18, help="#lambda points in sweep (logspace)")

    # 各 lam_true で何回シミュレーションするか
    ap.add_argument("--nrep", type=int, default=300, help="#replications per lambda")

    # 乱数シード（再現性のため）
    ap.add_argument("--seed", type=int, default=1, help="RNG seed")

    # 推定で使う lambda 範囲の上限（分岐比テーブルの上限にも使う）
    ap.add_argument("--rate_max", type=float, default=41.0, help="Upper bound for lambda estimation + branching grid")

    # 分岐比テーブル（rate_grid）の分割数：大きいほど補間誤差は減るが計算・メモリ増
    ap.add_argument("--rate_grid_n", type=int, default=20001, help="Grid size for calc_branchingratios(rate_grid)")

    # profile CI の brentq が失敗した場合のフォールバック用グリッド点数
    ap.add_argument("--scan_points", type=int, default=1500, help="Fallback profile scan points (logspace)")

    # グレードの「ラベル」列（intでもstrでもOKという実装）
    ap.add_argument("--itypenames", type=str, default="0,1,2,3,4", help="Comma-separated grade IDs (or names)")

    # lambda 推定に使うグレード（例：Hp/Mp/Ms）
    ap.add_argument("--fit_grades", type=str, default="0,1,2", help="Comma-separated grades used to fit lambda (Hp/Mp/Ms)")

    # 推定後に評価したいグレード（例：Lp/Ls）
    ap.add_argument("--target_grades", type=str, default="3,4", help="Comma-separated grades to evaluate (Lp/Ls)")

    # 理論曲線の確認プロットを出すか
    ap.add_argument("--plot_theory", action="store_true", help="Also plot theory expected counts and sum_p(lam)")

    args = ap.parse_args()

    def parse_list(s):
        """
        "0,1,2" のような文字列を [0,1,2] に変換する。
        int にできない場合は文字列のまま残す（名前ラベル対応のため）。
        """
        items = [x.strip() for x in s.split(",") if x.strip() != ""]
        out = []
        for x in items:
            try:
                out.append(int(x))
            except ValueError:
                out.append(x)
        return tuple(out)

    # グレード設定を読み取る
    itypenames = parse_list(args.itypenames)
    fit_grades = parse_list(args.fit_grades)
    target_grades = parse_list(args.target_grades)

    # p_of(g, lam) を構築（分岐比の補間関数）
    p_of = build_branching_p_of(args.rate_max, args.rate_grid_n, itypenames)

    # lam_true の掃引点：logspace（暗い〜明るいを桁で均等に）
    lam_grid = np.logspace(np.log10(args.lam_min), np.log10(args.lam_max), int(args.nlam))

    # 理論曲線を確認したい場合（実行前の sanity check）
    if args.plot_theory:
        plot_expected_counts(
            lam_grid=lam_grid,
            T=float(args.T),
            p_of=p_of,
            itypenames=itypenames,
            fit_grades=fit_grades,
            target_grades=target_grades,
            title_suffix=f"(T={args.T:g}s, fit={fit_grades}, target={target_grades})"
        )

    # シミュレーション掃引の実行
    summary = run_sweep(
        lam_grid=lam_grid,
        T=float(args.T),
        nrep=int(args.nrep),
        seed=int(args.seed),
        p_of=p_of,
        itypenames=itypenames,
        fit_grades=fit_grades,
        target_grades=target_grades,
        rate_max=float(args.rate_max),
        scan_points=int(args.scan_points),
    )

    # --- テキスト出力（簡易表） ---
    print("\nlam_true  ok_frac  lam_CIrel  " + "  ".join([f"t{g}_CIrel" for g in target_grades]))
    for r in summary:
        line = f"{r['lam_true']:.6g}  {r['ok_frac']:.6f}  {r['lam_ciw_rel_med']:.6g}"
        for g in target_grades:
            line += f"  {r[f'target_{g}_ciw_rel_med']:.6g}"
        print(line)

    # --- 図の出力 ---
    plot_summary(summary, target_grades)


if __name__ == "__main__":
    main()    