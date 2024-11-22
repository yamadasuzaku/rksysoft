import numpy as np
import matplotlib.pyplot as plt

# ファイルを読み込む関数
def read_ion_data(file_path):
    """
    Read the ion data from a text file.

    Parameters:
    file_path (str): Path to the input file.

    Returns:
    tuple: Depth array and ion data dictionary.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # ヘッダー行の処理
    header = lines[0].strip().split('\t')
    depth_label = header[0]
    ion_labels = header[1:]

    # データ行の処理
    data = np.loadtxt(lines[1:], delimiter='\t')
    depths = data[:, 0]  # 最初の列が深さ
    ion_data = {label: data[:, i + 1] for i, label in enumerate(ion_labels)}

    # Depth でソート
    sorted_indices = np.argsort(depths)
    depths = depths[sorted_indices]
    ion_data = {label: values[sorted_indices] for label, values in ion_data.items()}

    return depths, ion_data

# プロット関数
def plot_ion_data(depths, ion_data, output_file=None):
    """
    Plot ion data with dot-line style.

    Parameters:
    depths (array): Depth values.
    ion_data (dict): Dictionary of ion data.
    output_file (str): Optional path to save the plot.
    """
    plt.figure(figsize=(10, 6))

    # 各イオンをプロット
    for i, (ion, values) in enumerate(ion_data.items()):
#        if i > 10:
        plt.plot(depths, values, linestyle='-', marker='o', label=ion, ms=1)

    plt.xscale('log')
    plt.yscale('linear')
    plt.xlabel("Depth")
    plt.ylabel("Abundance")
    plt.title("Iron Ion Abundances")
    plt.legend(loc='upper right', fontsize=8, ncol=2)
    plt.grid(visible=True, which='both', linestyle='--', linewidth=0.5)

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    else:
        plt.show()

# ファイルのパスを指定
file_path = "cygx1_lhs_.fe"

# データを読み込み、プロット
depths, ion_data = read_ion_data(file_path)
plot_ion_data(depths, ion_data)
