#!/usr/bin/env python 

# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from astropy.time import Time
import os
import sys
import argparse

# データの読み込みとフィルタリング
def load_and_filter_data(filename, relative_error_threshold=10):
    # データを読み込み、NumPy配列に変換
    data = np.loadtxt(filename)

    # MJD列を抽出し、Timeオブジェクトに変換
    mjd = data[:, 0]
    time_objs = Time(mjd, format='mjd')
    dates = time_objs.to_datetime()  # MJDをdatetimeに変換

    # データ列をそれぞれ抽出
    flux_2_20keV =   np.abs(data[:, 1])
    err_2_20keV =    np.abs(data[:, 2])
    flux_2_4keV =    np.abs(data[:, 3])
    err_2_4keV =     np.abs(data[:, 4])
    flux_4_10keV =   np.abs(data[:, 5])
    err_4_10keV =    np.abs(data[:, 6])
    flux_10_20keV =  np.abs(data[:, 7])
    err_10_20keV =   np.abs(data[:, 8])

    # 相対誤差を計算し、30%以上のデータを除去
    x = relative_error_threshold / 100.0  # 相対誤差のしきい値（30%）
    relative_error_2_20keV = err_2_20keV / flux_2_20keV
    mask = relative_error_2_20keV <= x

    # マスクを用いてデータをフィルタリング
    filtered_data = {
        "dates": np.array(dates)[mask],
        "mjd": mjd[mask],  # MJDもフィルタリング
        "flux_2_20keV": flux_2_20keV[mask],
        "err_2_20keV": err_2_20keV[mask],
        "flux_2_4keV": flux_2_4keV[mask],
        "err_2_4keV": err_2_4keV[mask],
        "flux_4_10keV": flux_4_10keV[mask],
        "err_4_10keV": err_4_10keV[mask],
        "flux_10_20keV": flux_10_20keV[mask],
        "err_10_20keV": err_10_20keV[mask]
    }
    return filtered_data

# 比率と誤差の計算
def calculate_hardness_ratios(data):
    # 比率を計算
    ratio_4_10keV_2_4keV = data["flux_4_10keV"] / data["flux_2_4keV"]
    ratio_10_20keV_2_4keV = data["flux_10_20keV"] / data["flux_2_4keV"]

    # 比率の誤差を誤差伝播により計算
    ratio_err_4_10keV_2_4keV = ratio_4_10keV_2_4keV * np.sqrt((data["err_4_10keV"] / data["flux_4_10keV"])**2 + (data["err_2_4keV"] / data["flux_2_4keV"])**2)
    ratio_err_10_20keV_2_4keV = ratio_10_20keV_2_4keV * np.sqrt((data["err_10_20keV"] / data["flux_10_20keV"])**2 + (data["err_2_4keV"] / data["flux_2_4keV"])**2)

    return {
        "ratio_4_10keV_2_4keV": ratio_4_10keV_2_4keV,
        "ratio_err_4_10keV_2_4keV": ratio_err_4_10keV_2_4keV,
        "ratio_10_20keV_2_4keV": ratio_10_20keV_2_4keV,
        "ratio_err_10_20keV_2_4keV": ratio_err_10_20keV_2_4keV
    }

# Plotlyのプロットを作成
def create_plotly_subplots(dates, flux_data, ratio_data, base_filename, start_date=None, end_date=None, highlight_periods=None):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=False, vertical_spacing=0.1,
                        subplot_titles=('Time vs Flux', 'Time vs Hardenss', 'Flux vs Hardness'))

    fig.add_trace(go.Scatter(x=dates, y=flux_data["flux_2_20keV"], mode='markers',
                             name='2-20 keV',
                             error_y=dict(type='data', array=flux_data["err_10_20keV"], visible=True)),
                  row=1, col=1)

    fig.add_trace(go.Scatter(x=dates, y=flux_data["flux_2_4keV"], mode='markers',
                             name='2-4 keV',
                             error_y=dict(type='data', array=flux_data["err_2_4keV"], visible=True)),
                  row=1, col=1)

    fig.add_trace(go.Scatter(x=dates, y=flux_data["flux_4_10keV"], mode='markers',
                             name='4-10 keV',
                             error_y=dict(type='data', array=flux_data["err_4_10keV"], visible=True)),
                  row=1, col=1)

    fig.add_trace(go.Scatter(x=dates, y=flux_data["flux_10_20keV"], mode='markers',
                             name='10-20 keV',
                             error_y=dict(type='data', array=flux_data["err_10_20keV"], visible=True)),
                  row=1, col=1)

    fig.add_trace(go.Scatter(x=dates, y=ratio_data["ratio_10_20keV_2_4keV"], mode='markers',
                             name='10-20 keV / 2-4 keV',
                             error_y=dict(type='data', array=ratio_data["ratio_err_10_20keV_2_4keV"], visible=True)),
                  row=2, col=1)

    fig.add_trace(go.Scatter(x=dates, y=ratio_data["ratio_4_10keV_2_4keV"], mode='markers',
                             name='4-10 keV / 2-4 keV',
                             error_y=dict(type='data', array=ratio_data["ratio_err_4_10keV_2_4keV"], visible=True)),
                  row=2, col=1)


    fig.add_trace(go.Scatter(x=flux_data["flux_2_20keV"], y=ratio_data["ratio_4_10keV_2_4keV"], mode='markers',
                             name='Flux vs Hardness',
                             error_x=dict(type='data', array=flux_data["err_2_20keV"], visible=True),
                             error_y=dict(type='data', array=ratio_data["ratio_err_4_10keV_2_4keV"], visible=True)),
                  row=3, col=1)

    # 表示範囲の設定
    if start_date and end_date:
        fig.update_xaxes(range=[start_date, end_date], row=1, col=1)
        fig.update_xaxes(range=[start_date, end_date], row=2, col=1)

    # ハイライトの設定
    if highlight_periods:
        for start, end in highlight_periods:
            fig.add_vrect(x0=start, x1=end, fillcolor="orange", opacity=0.3, line_width=0, row=1, col=1)
            fig.add_vrect(x0=start, x1=end, fillcolor="orange", opacity=0.3, line_width=0, row=2, col=1)

    fig.update_layout(
        height=900,
        font_family="Times New Roman",
        title=f'Interactive Light Curve and Hardness Ratios ({base_filename})',
        xaxis_title='Date',
        xaxis2_title='Date',
        xaxis3_title='Flux [ph/s/cm²]',
        yaxis_title='Flux [ph/s/cm²]',
        yaxis2_title='hardness',
        yaxis3_title='4-10 keV / 2-4 keV',
        hovermode='x unified'
    )

    return fig

# Matplotlibのプロットを作成
def create_matplotlib_subplots(dates, flux_data, ratio_data, base_filename, start_date=None, end_date=None, highlight_periods=None):
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 8), sharex=False)

    axes[0].errorbar(dates, flux_data["flux_2_20keV"], yerr=flux_data["err_2_20keV"],
                     fmt='.', label='2-20 keV', color='red')
    axes[0].errorbar(dates, flux_data["flux_2_4keV"], yerr=flux_data["err_4_10keV"],
                     fmt='.', label='2-4 keV', color='green')
    axes[0].errorbar(dates, flux_data["flux_4_10keV"], yerr=flux_data["err_4_10keV"],
                     fmt='.', label='4-10 keV', color='cyan')
    axes[0].errorbar(dates, flux_data["flux_10_20keV"], yerr=flux_data["err_10_20keV"],
                     fmt='.', label='10-20 keV', color='blue')

    axes[0].set_ylabel('flux')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].errorbar(dates, ratio_data["ratio_10_20keV_2_4keV"], yerr=ratio_data["ratio_err_10_20keV_2_4keV"],
                     fmt='.', label='10-20 keV / 2-4 keV', color='red')
    axes[1].errorbar(dates, ratio_data["ratio_4_10keV_2_4keV"], yerr=ratio_data["ratio_err_4_10keV_2_4keV"],
                     fmt='.', label='4-10 keV / 2-4 keV', color='blue')

    axes[1].set_ylabel('hardness')
    axes[1].legend()
    axes[1].grid(True)

    axes[2].errorbar(flux_data["flux_2_20keV"], ratio_data["ratio_4_10keV_2_4keV"],
                     xerr=flux_data["err_2_20keV"], yerr=ratio_data["ratio_err_4_10keV_2_4keV"],
                     fmt='.', label='Flux vs Hardness', color='red')
    axes[2].set_xlabel('Flux [ph/s/cm²]')
    axes[2].set_ylabel('4-10 keV / 2-4 keV')
    axes[2].legend()
    axes[2].grid(True)

    # 表示範囲の設定
    if start_date and end_date:
        axes[0].set_xlim(start_date, end_date)
        axes[1].set_xlim(start_date, end_date)

    # ハイライトの設定
    if highlight_periods:
        for start, end in highlight_periods:
            axes[0].fill_between(dates, min(flux_data["flux_2_20keV"]) - 0.5, max(flux_data["flux_2_20keV"]) + 0.5,
                                 where=((dates >= start) & (dates <= end)),
                                 color='orange', alpha=0.3)
            axes[1].fill_between(dates, min(ratio_data["ratio_10_20keV_2_4keV"]) - 0.5, max(ratio_data["ratio_10_20keV_2_4keV"]) + 0.5,
                                 where=((dates >= start) & (dates <= end)),
                                 color='orange', alpha=0.3)

    plt.suptitle(f'Light Curve and Hardness Ratios ({base_filename})')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    return fig

# argparse設定
def get_args():
    parser = argparse.ArgumentParser(description="Plot light curve and hardness ratios using matplotlib or plotly.")
    parser.add_argument("filename", help="Path to the input data file")
    parser.add_argument("--plotter", choices=["matplotlib", "plotly"], default="matplotlib",
                        help="Choose the plotting library: 'matplotlib' or 'plotly'")
    parser.add_argument("--start_date", help="Start date in ISOT format (e.g., '2023-10-15T00:00:00')", default=None)
    parser.add_argument("--end_date", help="End date in ISOT format (e.g., '2024-05-01T00:00:00')", default=None)
    parser.add_argument("--highlight", nargs=2, action='append', metavar=('START', 'END'),
                        help="Highlight period in ISOT format (e.g., '2023-11-03T23:51:04')")
    return parser.parse_args()

# メイン処理
def main(filename, plot_library, start_date=None, end_date=None, highlight_periods=None):
    base_filename = os.path.splitext(os.path.basename(filename))[0]

    # データの読み込みとフィルタリング
    flux_data = load_and_filter_data(filename)

    # ハードネス比とその誤差の計算
    ratio_data = calculate_hardness_ratios(flux_data)

    # 時間範囲の処理
    if start_date:
        start_date = Time(start_date.strip(), format="isot").datetime  # stripで余計なスペースを削除
    if end_date:
        end_date = Time(end_date.strip(), format="isot").datetime

    # ハイライト期間の処理
    if highlight_periods:
        highlight_periods = [(Time(start.strip(), format="isot").datetime, Time(end.strip(), format="isot").datetime) for start, end in highlight_periods]

    if plot_library == 'plotly':
        fig = create_plotly_subplots(flux_data["dates"], flux_data, ratio_data, base_filename, start_date, end_date, highlight_periods)
        html_filename = f"{base_filename}_plotly.html"
        pio.write_html(fig, file=html_filename)
        fig.show()
    elif plot_library == 'matplotlib':
        fig = create_matplotlib_subplots(flux_data["dates"], flux_data, ratio_data, base_filename, start_date, end_date, highlight_periods)
        plt.savefig(f"{base_filename}_matplotlib.png", dpi=300)
        plt.show()

    if highlight_periods:
        # ハイライト部分の値を内挿で推定する例
        # datetimeのリストを数値に変換
        dates_num = np.array([date.timestamp() for date in flux_data["dates"]])
        # highlight_periodsをtimestampに変換
        highlight_times = np.array([date.timestamp() for period in highlight_periods for date in period])
        # numpy.interpを用いてhighlight_periodsの時間で内挿したflux_2_20keVの値を計算
        highlight_flux_2_20keV = np.interp(highlight_times, dates_num, flux_data["flux_2_20keV"])
        # 内挿結果の表示
        for i, period in enumerate(highlight_periods):
            print(f"Period {period[0]} to {period[1]}:")
            print(f"Interpolated flux at {period[0]}: {highlight_flux_2_20keV[2*i]}")
            print(f"Interpolated flux at {period[1]}: {highlight_flux_2_20keV[2*i+1]}")

# Google Colab向け: `argparse`を回避して直接値を指定できるようにする
def run_in_colab():

    # ファイルを指定
    filename = "J1958+352_g_lc_1day_all.dat"  # Colabにアップロードされたファイルを指定

    # プロッターを指定
#    plot_library = "matplotlib"
    plot_library = "plotly"

    # 描画範囲の時刻を指定
#    start_date = None
#    end_date = None
    start_date = "2023-11-01T00:00:00"
    end_date = "2024-06-01T00:00:00"

    # ハイライトしたい時刻範囲を指定
    highlight_periods = [
        ("2023-11-03T23:51:04", "2023-11-05T14:01:04"),
        ("2024-04-07T16:55:04", "2024-04-10T13:41:04")
    ]
    main(filename, plot_library, start_date, end_date, highlight_periods)

if __name__ == "__main__":
    # Google Colab環境の確認
    if "google.colab" in sys.modules:
        print("Running in Google Colab environment. Using predefined settings.")
        run_in_colab()
    else:
        args = get_args()  # スタンドアローンの場合、引数をパース
        main(args.filename, args.plotter, args.start_date, args.end_date, args.highlight)
