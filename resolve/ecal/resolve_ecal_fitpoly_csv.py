import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
params = {'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 8}
plt.rcParams['font.family'] = 'serif'
plt.rcParams.update(params)

from numpy.polynomial import Polynomial

# CSVファイルの読み込み
def read_csv(filename):
    print(f"Reading CSV file: {filename}")
    data = pd.read_csv(filename)
    print(f"Data read successfully. Columns: {data.columns}")
    return data

# 境界点boundaryで2つの多項式フィッティングを行う
def fit_piecewise_polynomial(x, y, boundary, degree1, degree2):
    print(f"Performing piecewise polynomial fit with boundary: {boundary}")
    
    # 境界点boundaryでデータを分割
    mask1 = x < boundary
    mask2 = x > boundary
    
    x1, y1 = x[mask1], y[mask1]
    x2, y2 = x[mask2], y[mask2]
    
    print(f"Data points for x < {boundary}: {len(x1)}")
    print(f"Data points for x > {boundary}: {len(x2)}")
    
    # 境界点boundaryでの値を追加
    if np.any(x == boundary):
        boundary_value = np.mean(y[x == boundary])
        print(f"Boundary value found: {boundary_value}")
    else:
        boundary_value = (y1[-1] + y2[0]) / 2
        print(f"Boundary value approximated: {boundary_value}")
        
    x1 = np.append(x1, boundary)
    y1 = np.append(y1, boundary_value)
    x2 = np.insert(x2, 0, boundary)
    y2 = np.insert(y2, 0, boundary_value)
    
    # 各部分に対して多項式フィッティング
    print("Fitting polynomial for x < boundary")
    p1 = Polynomial.fit(x1, y1, degree1)
    
    print("Fitting polynomial for x > boundary")
    p2 = Polynomial.fit(x2, y2, degree2)
    
    return p1, p2

# プロット
def plot_data_and_model(data, model1, model2, boundary, x_col, y_col, t_col, degree1, degree2, output_filename, xmin, xmax):
    x = data[x_col]
    y = data[y_col]
    time = data[t_col]
    
    # モデルによる予測値
    print("Generating model predictions")
    y_model1 = model1(x)
    y_model2 = model2(x)
    y_model = np.where(x < boundary, y_model1, y_model2)
    
    # データとモデルの比
    ratio = y / y_model
    
    # フィット範囲の制限
    fit_mask = (x >= xmin) & (x <= xmax)
    x_fit = x[fit_mask]
    y_fit = y[fit_mask]
    y_model_fit = y_model[fit_mask]
    ratio_fit = ratio[fit_mask]
    time_fit = time[fit_mask]
    
    # プロットの設定
    print("Creating plots"), 
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
    
    # 上のパネル：データとモデル
    ax1.plot(x_fit, y_fit, 'bo', label='Data')
    ax1.plot(x_fit, y_model_fit, 'r-', label=f'Piecewise Polynomial fit (degrees {degree1}, {degree2})')
    ax1.set_ylabel(y_col)
    ax1.set_xlim(xmin,xmax)
    ax1.legend()
    ax1.set_title('Data and Piecewise Polynomial Fit')
    
    # 下のパネル：データとモデルの比
    ax2.plot(x_fit, ratio_fit, 'go', label='Data / Model')
    ax2.set_xlabel(x_col)
    ax2.set_xlim(xmin,xmax)
    ax2.set_ylabel('Data / Model')
    ax2.legend()
    ax2.set_title('Data / Model Ratio')
    ax2.axhline(1.0, color='black', linestyle='--', linewidth=2, label='ratio = 1.0')
    plt.tight_layout()
    plt.savefig(output_filename)
    print(f"Plot saved to {output_filename}")
    plt.show()


    # プロットの設定
    print("Creating plots for TIME vs. Residuals"), 
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 8))
    
    # 上のパネル：データとモデル
    ax1.plot(time_fit, ratio_fit, 'bo', label='Ratio')
    ax1.set_ylabel(y_col)
    ax1.legend()
    ax1.set_title('Time history of residuals')
    plt.tight_layout()
    plt.savefig("resi_" + output_filename)
    print(f"Plot saved to resi_{output_filename}")
    plt.show()
    
    # CSVファイルにダンプ
    csv_filename = output_filename.replace(".png", ".csv")
    print(f"Dumping time_fit and ratio_fit to CSV file: {csv_filename}")
    df_fit = pd.DataFrame({t_col: time_fit, 'Ratio': ratio_fit})
    df_fit.to_csv(csv_filename, index=False)
    print(f"Data successfully dumped to {csv_filename}")

if __name__ == "__main__":
    import argparse

    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description='Fit a piecewise polynomial to CSV data and plot the results.')
    parser.add_argument('csvfile', type=str, help='Path to the CSV file')
    parser.add_argument('x_col', type=str, help='Column name for x data')
    parser.add_argument('y_col', type=str, help='Column name for y data')
    parser.add_argument('boundary', type=float, help='Boundary point for piecewise fitting')
    parser.add_argument('degree1', type=int, help='Degree of the polynomial fit for x < boundary')
    parser.add_argument('degree2', type=int, help='Degree of the polynomial fit for x > boundary')
    parser.add_argument('outputfile', type=str, help='Filename for the output plot')
    parser.add_argument('--xmin', type=float, help='Minimum x value for fitting', default=2000)
    parser.add_argument('--xmax', type=float, help='Maximum x value for fitting', default=40000)
    parser.add_argument('--t_col', type=str, help='Column name for TIME', default="TIME")
    args = parser.parse_args()
    
    # CSVファイルの読み込み
    data = read_csv(args.csvfile)
    
    # データのソート
    data = data.sort_values(by=args.x_col)
    print(f"Data sorted by {args.x_col}")
    
    # フィット範囲の設定
    xmin = args.xmin if args.xmin is not None else data[args.x_col].min()
    xmax = args.xmax if args.xmax is not None else data[args.x_col].max()
    print(f"Fitting range: {xmin} to {xmax}")
    
    # フィット範囲内のデータを抽出
    fit_mask = (data[args.x_col] >= xmin) & (data[args.x_col] <= xmax)
    data_fit = data[fit_mask]
    
    # 境界点boundaryで2つの多項式フィッティング
    x = data_fit[args.x_col].values
    y = data_fit[args.y_col].values
    time = data_fit[args.t_col].values

    model1, model2 = fit_piecewise_polynomial(x, y, args.boundary, args.degree1, args.degree2)
    
    # プロット
    plot_data_and_model(data, model1, model2, args.boundary, args.x_col, args.y_col, args.t_col, args.degree1, args.degree2, args.outputfile, xmin, xmax)