#!/usr/bin/env python

import numpy as np
from astropy.io import fits
from astropy.time import Time
import matplotlib.pyplot as plt

# 擬似的な FITS ファイルを生成する関数
def create_mock_fits(filename="mock_fits.fits"):
    # 観測時間データ（MJDREFからの秒数で仮定）
    nrows = 100  # 行数（観測点の数）
    time_data = np.linspace(0, 500000, nrows)  # 仮の時間データ（秒）

    # MJDREF (基準時刻) を仮定
    mjdref = 55000.0  # 仮の基準MJD

    # FITSヘッダーとデータを作成
    col1 = fits.Column(name='TIME', format='E', array=time_data)
    cols = fits.ColDefs([col1])
    hdu1 = fits.BinTableHDU.from_columns(cols)

    # プライマリヘッダーにMJDREFを追加
    hdu0 = fits.PrimaryHDU()
    hdu0.header['MJDREF'] = mjdref

    # FITSファイルを作成
    hdul = fits.HDUList([hdu0, hdu1])
    hdul.writeto(filename, overwrite=True)

# FITSファイルを生成
create_mock_fits()

# FITSファイルを読み込んで位相を計算する関数
def calculate_orbital_phase(fits_filename):
    # FITSファイルを開く
    hdu = fits.open(fits_filename)

    # 観測時間列を取得 (MJDREF からの秒数)
    time = hdu[1].data["TIME"]

    # MJDREF (基準時刻) を取得
    mjdref = hdu[0].header["MJDREF"]

    # 基準時刻を astropy の Time オブジェクトに変換
    t_mjdref = Time(mjdref, format='mjd')

    # 観測時間を MJD に変換
    t_mjd = Time(t_mjdref.jd + time / 86400., format='mjd')

    # 軌道位相の計算
    To = 5.599829  # 軌道周期 (日)
    Do = 41874.207 # 基準時刻 (MJD)

    phase = []
    for onetime in t_mjd.mjd:
        # 位相を計算してリストに追加
        onephase = (onetime - Do) % To / To  # 0-1 の範囲に収める
        phase.append(onephase)

    # 計算結果をプロット
    plt.plot(t_mjd.mjd, phase, 'o-', label="orbital phase")
    plt.xlabel("MJD")
    plt.ylabel("Phase (0-1)")
    plt.title("Orbital Phase of Cyg X-1")
    plt.legend()
    plt.savefig("phase_demo_cyg.png")    
    plt.show()

# 軌道位相の計算とプロット
calculate_orbital_phase("mock_fits.fits")
