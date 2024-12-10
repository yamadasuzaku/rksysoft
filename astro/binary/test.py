import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse

# 定数
G = 6.67430e-11  # 重力定数 (m^3 kg^-1 s^-2)
M_sun = 1.989e30  # 太陽質量 (kg)
R_sun = 6.955e8   # 太陽半径 (m)

def calculate_orbital_period(distance, M1, M2):
    """Keplerの法則で公転周期を計算"""
    return 2 * np.pi * np.sqrt((distance**3) / (G * (M1 + M2)))

def calculate_positions(t, distance, M1, M2, orbital_period):
    """各時間で星の位置を計算"""
    omega = 2 * np.pi / orbital_period  # 角速度
    x1 = -(M2 / (M1 + M2)) * distance * np.cos(omega * t)
    y1 = -(M2 / (M1 + M2)) * distance * np.sin(omega * t)
    x2 = (M1 / (M1 + M2)) * distance * np.cos(omega * t)
    y2 = (M1 / (M1 + M2)) * distance * np.sin(omega * t)
    return x1, y1, x2, y2

def main(M1, M2, R1, R2, distance, steps):
    """アニメーションを生成"""
    # 公転周期の計算
    orbital_period = calculate_orbital_period(distance, M1, M2)

    # 時間ステップ
    time = np.linspace(0, orbital_period, steps)

    # アニメーションの作成
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-1.5 * distance, 1.5 * distance)
    ax.set_ylim(-1.5 * distance, 1.5 * distance)
    ax.set_aspect('equal')
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("近接連星系のアニメーション")

    star1, = ax.plot([], [], 'ro', markersize=10, label="Star 1 (M1)")
    star2, = ax.plot([], [], 'bo', markersize=10, label="Star 2 (M2)")
    orbit1, = ax.plot([], [], 'r--', linewidth=1, alpha=0.6)
    orbit2, = ax.plot([], [], 'b--', linewidth=1, alpha=0.6)

    # アニメーションの初期化
    def init():
        star1.set_data([], [])
        star2.set_data([], [])
        orbit1.set_data([], [])
        orbit2.set_data([], [])
        return star1, star2, orbit1, orbit2

    # フレームの更新
    def update(frame):
        t = time[frame]
        x1, y1, x2, y2 = calculate_positions(t, distance, M1, M2, orbital_period)

        star1.set_data(x1, y1)
        star2.set_data(x2, y2)

        orbit1.set_data(-(M2 / (M1 + M2)) * distance * np.cos(2 * np.pi * time / orbital_period),
                        -(M2 / (M1 + M2)) * distance * np.sin(2 * np.pi * time / orbital_period))
        orbit2.set_data((M1 / (M1 + M2)) * distance * np.cos(2 * np.pi * time / orbital_period),
                        (M1 / (M1 + M2)) * distance * np.sin(2 * np.pi * time / orbital_period))
        return star1, star2, orbit1, orbit2

    ani = FuncAnimation(fig, update, frames=len(time), init_func=init, blit=True, interval=30)

    # アニメーションの保存（必要ならコメントアウトを解除）
    # ani.save('binary_star_system.mp4', fps=30, writer='ffmpeg')

    plt.legend()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="近接連星系のアニメーションを生成")
    parser.add_argument("--M1", type=float, default=10 * M_sun, help="星1の質量 (kg)")
    parser.add_argument("--M2", type=float, default=10 * M_sun, help="星2の質量 (kg)")
    parser.add_argument("--R1", type=float, default=10 * R_sun, help="星1の半径 (m)")
    parser.add_argument("--R2", type=float, default=10 * R_sun, help="星2の半径 (m)")
    parser.add_argument("--distance", type=float, default=10 * R_sun, help="星同士の距離 (m)")
    parser.add_argument("--steps", type=int, default=1000, help="アニメーションの時間ステップ数")
    
    args = parser.parse_args()

    main(args.M1, args.M2, args.R1, args.R2, args.distance, args.steps)
