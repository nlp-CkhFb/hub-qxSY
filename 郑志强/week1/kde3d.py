# -*- coding: utf-8 -*-
"""
三维核密度曲线（3D Kernel Density）绘制工具
==========================================

"三维核密度"通常有三种画法，本脚本一次性全部实现，你按需取用：

  1. surface    密度曲面 —— KDE 在 (x, y) 网格上的密度值作为 z 高度，画成 3D 曲面
                 （最常说的"三维核密度曲线"就是这种：双变量 KDE 的立体呈现）
  2. volume     密度等值面 —— 频数作为第三个维度，用 marching cubes 抽出 3D 等值面，
                 真正在三维空间里看联合密度
  3. projection 投影密度峰 —— 散点铺在 z=0 平面，每个点的高度等于该处核密度值

核密度估计公式（高斯核）：

    f(x) = 1/(n·h^d·(2π)^(d/2)) · Σ_i exp( -||x - x_i||² / (2h²) )

    d 维数，n 样本数，h 带宽(bandwidth)——h 越小越尖锐、越大越平滑。
    带宽由 Scott 规则自动选取：h = n^(-1/(d+4))

用法：
    python kde3d.py                       # 用内置模拟数据画全部三种图
    python kde3d.py --data mydata.csv     # 用你自己的 CSV（取前 3 个数值列）
    python kde3d.py --kind surface --bw 0.25
    python kde3d.py --kind volume --levels 5
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
# 无界面环境下自动退到 Agg，避免 plt.show() 报错
if not os.environ.get('DISPLAY') and sys.platform.startswith('linux'):
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors, font_manager
from scipy.stats import gaussian_kde

# --------------------------------------------------------------------------
# 全局绘图设置：中文字体（否则标题/坐标轴中文会变成方块）
# --------------------------------------------------------------------------
_CJK_CANDIDATES = ['Microsoft YaHei', 'SimHei', 'DengXian', 'SimSun',
                   'Noto Sans CJK SC', 'Source Han Sans SC', 'PingFang SC',
                   'Heiti SC', 'Arial Unicode MS']


def setup_chinese_font(verbose: bool = True) -> str:
    """挑一个系统里存在的中文字体，返回字体名。"""
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in _CJK_CANDIDATES:
        if name in available:
            plt.rcParams['font.sans-serif'] = [name] + plt.rcParams['font.sans-serif']
            plt.rcParams['axes.unicode_minus'] = False      # 负号正常显示
            if verbose:
                print('使用中文字体：%s' % name)
            return name
    if verbose:
        print('警告：未找到中文字体，图中中文可能显示为方块')
    return ''


def _thin_zticks(ax, n: int = 6):
    """matplotlib 3.4 的 3D 轴刻度较密，会把 z 轴标签挤到重叠，这里做抽稀。"""
    try:
        ticks = list(ax.get_zticks())
        if len(ticks) > n:
            ax.set_zticks(ticks[::max(1, len(ticks) // n)])
    except Exception:
        pass


# --------------------------------------------------------------------------
# 数据
# --------------------------------------------------------------------------
def make_sample_data(n: int = 1200, seed: int = 42) -> np.ndarray:
    """生成双峰带相关性的三维模拟数据 (n, 3)，用来演示核密度。"""
    rng = np.random.default_rng(seed)
    n1, n2 = n // 2, n - n // 2

    # 峰 1：正相关
    cov1 = np.array([[1.0, 0.75, 0.45],
                     [0.75, 1.0, 0.35],
                     [0.45, 0.35, 1.0]]) * 0.55
    c1 = rng.multivariate_normal([-1.4, -1.0, 0.3], cov1, n1)

    # 峰 2：负相关
    cov2 = np.array([[1.0, -0.65, 0.20],
                     [-0.65, 1.0, -0.30],
                     [0.20, -0.30, 1.0]]) * 0.45
    c2 = rng.multivariate_normal([1.5, 1.2, -0.4], cov2, n2)

    return np.vstack([c1, c2])


def load_data(path: str, use_cols=None):
    """从 CSV 读取数据：取数值列，返回 (data, labels)。"""
    import pandas as pd
    df = pd.read_csv(path)
    num = df.select_dtypes(include=[np.number])
    # 丢掉常数列（核密度需要方差）
    num = num.loc[:, num.std() > 0]
    if use_cols:
        num = num[[c for c in use_cols if c in num.columns]]
    if num.shape[1] < 2:
        raise SystemExit('CSV 里至少需要 2 个数值列才能画三维核密度')
    cols = list(num.columns[:3])
    labels = cols + ['密度'] * (3 - len(cols))
    return num[cols].dropna().to_numpy(), labels


# --------------------------------------------------------------------------
# 核密度估计
# --------------------------------------------------------------------------
def kde_on_grid(data: np.ndarray, bw: float = None, grid: int = 90, pad: float = 0.6):
    """在 (x, y) 网格上评估二维高斯核密度。

    返回 (X, Y, Z, kde)，Z 为密度值矩阵。
    """
    d = data[:, :2].T
    kde = gaussian_kde(d, bw_method=bw)          # bw=None 时用 Scott 规则

    xmin, ymin = data[:, 0].min(), data[:, 1].min()
    xmax, ymax = data[:, 0].max(), data[:, 1].max()
    xs = np.linspace(xmin - pad, xmax + pad, grid)
    ys = np.linspace(ymin - pad, ymax + pad, grid)
    X, Y = np.meshgrid(xs, ys)
    pos = np.vstack([X.ravel(), Y.ravel()])
    Z = kde(pos).reshape(X.shape)
    return X, Y, Z, kde


# --------------------------------------------------------------------------
# 画法 1：密度曲面
# --------------------------------------------------------------------------
def plot_surface(data, labels=('X', 'Y', '密度'), bw=None, grid=90,
                 cmap='viridis', gray_scatter=False):
    """把二维核密度值当高度，画成 3D 曲面 + 底部散点。"""
    X, Y, Z, _ = kde_on_grid(data, bw=bw, grid=grid)
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 密度曲面（降采样一点，避免网格线过密）
    surf = ax.plot_surface(X, Y, Z, cmap=cmap, linewidth=0, antialiased=True,
                           alpha=0.92, rstride=1, cstride=1,
                           vmin=Z.min(), vmax=Z.max())
    # 投影到底面，帮助判断峰的位置
    ax.contour(X, Y, Z, zdir='z', offset=Z.min() - 0.05 * (Z.max() - Z.min()),
               cmap=cmap, levels=12, linewidths=0.8, alpha=0.75)

    # 原始样本点落在底面
    z0 = Z.min() - 0.05 * (Z.max() - Z.min())
    ax.scatter(data[:, 0], data[:, 1], np.full(len(data), z0),
               s=6, c='0.35' if gray_scatter else data[:, 2],
               cmap=('Greys' if gray_scatter else cmap),
               alpha=0.45, depthshade=False, label='样本点 n=%d' % len(data))

    fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.09, label=labels[2])
    ax.set_xlabel(labels[0], labelpad=10, fontsize=11)
    ax.set_ylabel(labels[1], labelpad=10, fontsize=11)
    ax.set_zlabel(labels[2], labelpad=8, fontsize=11)
    ax.set_title('三维核密度曲面（高斯核，带宽=%s）'
                 % ('Scott' if bw is None else bw), fontsize=14, pad=18)
    ax.view_init(elev=28, azim=-58)
    _thin_zticks(ax)
    ax.legend(loc='upper left', fontsize=9)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# 画法 2：三维等值面（真正的 3D 密度）
# --------------------------------------------------------------------------
def plot_volume(data, labels=('X', 'Y', 'Z'), bw=None, grid=46, levels=4,
                cmap='Spectral_r', alpha=0.32, view=(22, -60)):
    """用 marching cubes 从三维核密度里抽出等值面。

    这里第三个维度不再是密度值，而是数据的第 3 个特征；
    密度是四维量，用一组嵌套的等值面表示它的"浓度"。
    """
    from skimage import measure
    from sklearn.neighbors import KernelDensity

    if data.shape[1] < 3:
        raise SystemExit('等值面画法需要 3 列数值数据')

    pts = data[:, :3]
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    pad = 0.35 * (hi - lo)
    lo, hi = lo - pad, hi + pad

    # 自动带宽（Scott 规则）
    h = bw if bw is not None else len(pts) ** (-1.0 / (pts.shape[1] + 4))
    kde = KernelDensity(kernel='gaussian', bandwidth=h).fit(pts)

    gx = np.linspace(lo[0], hi[0], grid)
    gy = np.linspace(lo[1], hi[1], grid)
    gz = np.linspace(lo[2], hi[2], grid)
    GX, GY, GZ = np.meshgrid(gx, gy, gz, indexing='ij')
    positions = np.vstack([GX.ravel(), GY.ravel(), GZ.ravel()]).T
    dens = np.exp(kde.score_samples(positions)).reshape(GX.shape)

    # 取密度分位数作为等值面高度，保证每层里都有实体
    dmax = dens.max()
    iso = dmax * np.linspace(0.12, 0.85, levels)

    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111, projection='3d')
    cmap_obj = plt.get_cmap(cmap)
    norm = colors.Normalize(vmin=iso.min(), vmax=iso.max())

    drawn = 0
    handles, texts = [], []
    for lv in iso:
        try:
            verts, faces, _, _ = measure.marching_cubes(dens, level=lv,
                                                        spacing=(gx[1] - gx[0],
                                                                 gy[1] - gy[0],
                                                                 gz[1] - gz[0]))
        except (ValueError, RuntimeError):
            continue        # 该密度层没有闭合等值面
        verts = verts + lo
        ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2],
                        color=cmap_obj(norm(lv)), alpha=alpha,
                        linewidth=0, shade=True)
        drawn += 1
        # matplotlib 3.4 的 Poly3DCollection 不能直接进图例，用色块代理
        handles.append(plt.Line2D([], [], marker='s', linestyle='none',
                                  markersize=9, color=cmap_obj(norm(lv))))
        texts.append('密度 ≥ %.4f' % lv)

    if drawn == 0:
        print('提示：等值面层数没画出来，试试 --levels 更小或 --grid 更大')

    # 少量原始点做参考
    idx = np.random.default_rng(0).choice(len(pts), size=min(300, len(pts)),
                                          replace=False)
    ax.scatter(pts[idx, 0], pts[idx, 1], pts[idx, 2], s=4, c='0.25',
               alpha=0.25, depthshade=False)
    handles.append(plt.Line2D([], [], marker='o', linestyle='none',
                              markersize=5, color='0.25', alpha=0.4))
    texts.append('样本点 n=%d' % len(pts))

    ax.set_xlabel(labels[0], labelpad=8)
    ax.set_ylabel(labels[1], labelpad=8)
    ax.set_zlabel(labels[2], labelpad=6)
    ax.set_title('三维核密度等值面（嵌套层表示密度浓度，带宽=%.3f）' % h,
                 fontsize=14, pad=16)
    ax.view_init(elev=view[0], azim=view[1])
    ax.legend(handles, texts, loc='upper left', fontsize=8)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# 画法 3：投影密度峰
# --------------------------------------------------------------------------
def plot_projection(data, labels=('X', 'Y', '强度'), bw=None, grid=140,
                    cmap='magma', view=(24, -52), marker_size=55):
    """散点铺在底面，z 高度 = 每个样本点处的核密度值。"""
    _, _, _, kde = kde_on_grid(data, bw=bw, grid=grid)
    dens = kde(data[:, :2].T)
    dens = dens / dens.max()

    order = np.argsort(dens)                      # 低密度先画，高密度压在上面
    xs, ys, zs, cs = data[order, 0], data[order, 1], dens[order], dens[order]

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(xs, ys, zs, c=cs, s=marker_size, cmap=cmap,
                    alpha=0.72, depthshade=True, edgecolors='none')

    # 底部等密度线
    X, Y, Z, _ = kde_on_grid(data, bw=bw, grid=80)
    ax.contour(X, Y, Z / Z.max(), zdir='z', offset=0, cmap=cmap,
               levels=10, linewidths=0.8, alpha=0.7)

    fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.09, label='归一化核密度')
    ax.set_xlabel(labels[0], labelpad=10)
    ax.set_ylabel(labels[1], labelpad=10)
    ax.set_zlabel('核密度强度', labelpad=8)
    ax.set_title('三维核密度投影峰（高度=该点密度，带宽=%s）'
                 % ('Scott' if bw is None else bw), fontsize=14, pad=16)
    ax.view_init(elev=view[0], azim=view[1])
    _thin_zticks(ax)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# 汇总输出
# --------------------------------------------------------------------------
def save_all(fig, outdir: str, name: str, show: bool = False, dpi: int = 150):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, name + '.png')
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print('已保存 %s' % path)
    return path


def main():
    ap = argparse.ArgumentParser(description='三维核密度曲线绘制')
    ap.add_argument('--kind', default='all',
                    choices=['all', 'surface', 'volume', 'projection'],
                    help='画哪种：surface 密度曲面 / volume 等值面 / projection 投影峰')
    ap.add_argument('--data', default=None, help='CSV 文件路径；缺省用内置模拟数据')
    ap.add_argument('--cols', default=None, help='指定使用的列名，逗号分隔（最多 3 列）')
    ap.add_argument('--n', type=int, default=1200, help='模拟数据样本量')
    ap.add_argument('--bw', type=float, default=None,
                    help='核带宽；缺省用 Scott 规则自动选取。调小更尖锐，调大更平滑')
    ap.add_argument('--grid', type=int, default=90, help='surface/projection 的网格分辨率')
    ap.add_argument('--levels', type=int, default=4, help='volume 的等值面层数')
    ap.add_argument('--out', default='output_kde', help='输出目录')
    ap.add_argument('--show', action='store_true', help='同时弹出窗口显示')
    args = ap.parse_args()

    setup_chinese_font()

    if args.data:
        data, labels = load_data(args.data,
                                 args.cols.split(',') if args.cols else None)
        print('从 %s 读入 %d 个样本，使用列：%s' % (args.data, len(data), labels[:data.shape[1]]))
    else:
        data = make_sample_data(args.n)
        labels = ['特征 X', '特征 Y', '特征 Z']
        print('使用内置模拟数据：%d 个样本，%d 维' % (len(data), data.shape[1]))

    made = []
    if args.kind in ('all', 'surface'):
        fig = plot_surface(data, labels, bw=args.bw, grid=args.grid)
        made.append(save_all(fig, args.out, 'kde3d_surface', args.show))
    if args.kind in ('all', 'volume'):
        try:
            fig = plot_volume(data, labels, bw=args.bw, levels=args.levels)
            made.append(save_all(fig, args.out, 'kde3d_volume', args.show))
        except ImportError:
            print('跳过等值面画法：需要 scikit-image（pip install scikit-image）')
    if args.kind in ('all', 'projection'):
        fig = plot_projection(data, labels, bw=args.bw, grid=args.grid)
        made.append(save_all(fig, args.out, 'kde3d_projection', args.show))

    if args.show:
        plt.show()
    print('\n完成，共 %d 张图' % len(made))


if __name__ == '__main__':
    main()
