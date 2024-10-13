#!/usr/bin/env python 

# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
params = {'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 10}
plt.rcParams['font.family'] = 'serif'
plt.rcParams.update(params)

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from astropy.time import Time
import os
import sys
import argparse
from astropy.io import fits

# データの読み込みとフィルタリング
def load_and_filter_data(filename, relative_error_threshold=10):
    # データを読み込み、NumPy配列に変換
    with fits.open(filename) as hdul:
        # BINTABLE EXTNAME 'RATE' のデータを取得
        data = hdul[1].data
        # TIME, RATE, ERROR コラムを抽出
        mjd = data['TIME']
        rate = data['RATE']
        error = data['ERROR']
        # TIME (MJD) を datetime 形式に変換
        time_objs = Time(mjd, format='mjd')
        dates = time_objs.to_datetime()

    # numpy 配列に変換
    mjd = np.array(mjd)
    time_objs = np.array(time_objs)
    dates = np.array(dates)
    rate = np.array(rate)
    error = np.array(error)

    # 相対誤差を計算し、30%以上のデータを除去
    x = relative_error_threshold / 100.0  # 相対誤差のしきい値（30%）
    relative_error = error / rate
    mask = relative_error <= x

    # マスクを用いてデータをフィルタリング
    filtered_data = {
        "dates": dates[mask],
        "mjd": mjd[mask],  # MJDもフィルタリング
        "rate": rate[mask],
        "error": error[mask],
    }
    return filtered_data

# Plotlyのプロットを作成
def create_plotly_subplots(dates, flux_data, base_filename, start_date=None, end_date=None, highlight_periods=None):

    fig = make_subplots(rows=1, cols=1, shared_xaxes=False, vertical_spacing=0.1, subplot_titles=('Time vs Flux'))

    fig.add_trace(go.Scatter(x=dates, y=flux_data["rate"], mode='markers', name='15-50 keV',
                             error_y=dict(type='data', array=flux_data["error"], visible=True)), row=1, col=1)

    # 表示範囲の設定
    if start_date and end_date:
        fig.update_xaxes(range=[start_date, end_date], row=1, col=1)

    # ハイライトの設定
    if highlight_periods:
        for start, end in highlight_periods:
            print(f"..... highlight from {start} to {end}.")
            fig.add_vrect(x0=start, x1=end, fillcolor="orange", opacity=0.3, line_width=0, row=1, col=1)

    fig.update_layout(
        height=900,
        font_family="Times New Roman",
        title=f'Interactive Light Curve ({base_filename})',
        xaxis_title='Date',
        yaxis_title='Counts [/cm^2/s] (15-50 keV)',
        hovermode='x unified'
    )

    return fig

# Matplotlibのプロットを作成
def create_matplotlib_subplots(dates, flux_data, base_filename, start_date=None, end_date=None, highlight_periods=None):
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.errorbar(dates, flux_data["rate"], yerr=flux_data["error"],
                     fmt='.', label='15-50 keV', color='red')
    ax.set_ylabel('flux')
    ax.legend()
    ax.grid(True)

    # 表示範囲の設定
    if start_date and end_date:
        ax.set_xlim(start_date, end_date)

    # ハイライトの設定
    if highlight_periods:
        for start, end in highlight_periods:
            ax.fill_between(dates, min(flux_data["rate"]) - 0.5, max(flux_data["rate"]) + 0.5,
                                 where=((dates >= start) & (dates <= end)),
                                 color='orange', alpha=0.3)

    plt.suptitle(f'Light Curve ({base_filename})')
    plt.tight_layout()

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

    # 時間範囲の処理
    if start_date:
        start_date = Time(start_date.strip(), format="isot").datetime  # stripで余計なスペースを削除
    if end_date:
        end_date = Time(end_date.strip(), format="isot").datetime

    # ハイライト期間の処理

    if highlight_periods:
        highlight_periods = [(Time(start.strip(), format="isot").datetime, Time(end.strip(), format="isot").datetime) for start, end in highlight_periods]

    if plot_library == 'plotly':
        fig = create_plotly_subplots(flux_data["dates"], flux_data, base_filename, start_date, end_date, highlight_periods)
        html_filename = f"{base_filename}_plotly.html"
        pio.write_html(fig, file=html_filename)
        fig.show()
    elif plot_library == 'matplotlib':
        fig = create_matplotlib_subplots(flux_data["dates"], flux_data, base_filename, start_date, end_date, highlight_periods)
        plt.savefig(f"{base_filename}_matplotlib.png", dpi=300)
        plt.show()

    if highlight_periods:
        # ハイライト部分の値を内挿で推定する例
        # datetimeのリストを数値に変換
        dates_num = np.array([date.timestamp() for date in flux_data["dates"]])
        # highlight_periodsをtimestampに変換
        highlight_times = np.array([date.timestamp() for period in highlight_periods for date in period])
        # numpy.interpを用いてhighlight_periodsの時間で内挿したflux_2_20keVの値を計算
        highlight_rate = np.interp(highlight_times, dates_num, flux_data["rate"])
        # 内挿結果の表示
        for i, period in enumerate(highlight_periods):
            print(f"Period {period[0]} to {period[1]}:")
            print(f"Interpolated flux at {period[0]}: {highlight_rate[2*i]}")
            print(f"Interpolated flux at {period[1]}: {highlight_rate[2*i+1]}")

# Google Colab向け: `argparse`を回避して直接値を指定できるようにする
def run_in_colab():

    # ファイルを指定
    filename = "CygX-1.lc.fits"  # Colabにアップロードされたファイルを指定

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
