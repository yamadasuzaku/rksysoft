import numpy as np
import matplotlib.pyplot as plt

def plot_binary_system_with_velocity(
    stellar_radius=20.0,  # 恒星半径 (太陽半径単位)
    orbital_distance=40.0,  # 恒星とブラックホールの距離 (太陽半径単位)
    grid_size=50,  # 描画グリッドの分割数
    grid_range=60.0,  # 描画範囲（x, y方向の±値）
    v_scale=0.05,  # 矢印のスケール調整
    cmap='coolwarm'  # カラーマップ
):
    """
    連星系と速度場を可視化する。
    矢印の色を速度の大きさに応じて変化させ、colorbar を表示。

    Parameters:
        stellar_radius (float): 恒星の半径
        orbital_distance (float): 恒星とブラックホールの距離
        grid_size (int): 描画グリッドの分割数
        grid_range (float): 描画範囲（±値）
        v_scale (float): 矢印のスケール調整
        cmap (str): カラーマップの名前
    """
    # グリッド作成
    x = np.linspace(-grid_range, grid_range, grid_size)
    y = np.linspace(-grid_range, grid_range, grid_size)
    X, Y = np.meshgrid(x, y)

    # 恒星とブラックホールの位置
    stellar_x, stellar_y = 0, 0
    bh_x, bh_y = orbital_distance, 0

    # 恒星からの距離 r を計算
    R = np.sqrt((X - stellar_x)**2 + (Y - stellar_y)**2)

    # 球対称の速度場（v ∝ r）
    Vx = (X - stellar_x) / R  # x方向速度
    Vy = (Y - stellar_y) / R  # y方向速度
    Vx[R < stellar_radius] = 0  # 恒星内部では速度を0に設定
    Vy[R < stellar_radius] = 0

    # 速度のスケール調整
    Vx *= R * v_scale
    Vy *= R * v_scale

    # 視線方向の速度成分（射影スカラー値）
    n_x = X - bh_x
    n_y = Y - bh_y
    n_mag = np.sqrt(n_x**2 + n_y**2)
    n_x /= n_mag
    n_y /= n_mag
    V_parallel_magnitude = Vx * n_x + Vy * n_y

    # 全体の速度の大きさを計算
    V_magnitude = np.sqrt(Vx**2 + Vy**2)

    # 描画
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))

    # 視線方向成分のプロット
    quiver1 = axs[0].quiver(
        X, Y, Vx, Vy, V_parallel_magnitude,
        cmap=cmap, scale=1, scale_units='xy', angles='xy'
    )
    axs[0].plot(stellar_x, stellar_y, 'ro', markersize=10, label='Star (HDE 226868)')  # 恒星
    axs[0].plot(bh_x, bh_y, 'ko', markersize=10, label='Black Hole (Cyg X-1)')  # ブラックホール
    axs[0].set_xlim(-grid_range, grid_range)
    axs[0].set_ylim(-grid_range, grid_range)
    axs[0].set_aspect('equal', adjustable='datalim')
    axs[0].set_title('Line-of-Sight Velocity (Color by Magnitude)')
    axs[0].set_xlabel('X (arbitrary units)')
    axs[0].set_ylabel('Y (arbitrary units)')
    axs[0].legend()
    axs[0].grid()

    # Colorbar for line-of-sight component
    cbar1 = fig.colorbar(quiver1, ax=axs[0])
    cbar1.set_label('Line-of-Sight Velocity Magnitude')

    # 全体速度の大きさをカラーマップに反映
    quiver2 = axs[1].quiver(
        X, Y, Vx, Vy, V_magnitude,
        cmap=cmap, scale=1, scale_units='xy', angles='xy'
    )
    axs[1].plot(stellar_x, stellar_y, 'ro', markersize=10, label='Star (HDE 226868)')  # 恒星
    axs[1].plot(bh_x, bh_y, 'ko', markersize=10, label='Black Hole (Cyg X-1)')  # ブラックホール
    axs[1].set_xlim(-grid_range, grid_range)
    axs[1].set_ylim(-grid_range, grid_range)
    axs[1].set_aspect('equal', adjustable='datalim')
    axs[1].set_title('Velocity Magnitude (Color by Total Speed)')
    axs[1].set_xlabel('X (arbitrary units)')
    axs[1].set_ylabel('Y (arbitrary units)')
    axs[1].legend()
    axs[1].grid()

    # Colorbar for total velocity magnitude
    cbar2 = fig.colorbar(quiver2, ax=axs[1])
    cbar2.set_label('Total Velocity Magnitude')

    plt.tight_layout()
    plt.show()

# 実行 (HDE 226868 と Cyg X-1 を想定)
plot_binary_system_with_velocity()
