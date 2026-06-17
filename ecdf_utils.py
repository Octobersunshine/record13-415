import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import io
import base64
import os


_FONT_PATH = None
_FONT_PROP = None


def _get_font_prop():
    global _FONT_PROP, _FONT_PATH
    if _FONT_PROP is not None:
        return _FONT_PROP
    candidates = [
        'C:/Windows/Fonts/msyh.ttc',
        'C:/Windows/Fonts/simhei.ttf',
        'C:/Windows/Fonts/simsun.ttc',
    ]
    for p in candidates:
        if os.path.exists(p):
            _FONT_PATH = p
            _FONT_PROP = FontProperties(fname=p)
            return _FONT_PROP
    _FONT_PROP = FontProperties()
    return _FONT_PROP


def compute_ecdf(data):
    arr = np.asarray(data, dtype=float)
    arr = arr[~np.isnan(arr)]
    arr = np.sort(arr)
    n = len(arr)
    if n == 0:
        return np.array([]), np.array([])
    y = np.arange(1, n + 1) / n
    return arr, y


_PLOT_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
    '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
    '#bcbd22', '#17becf',
]


def _calc_visual_weights(counts, alpha=None):
    if not counts:
        return []
    max_n = max(counts)
    min_n = min(counts)
    weights = []
    range_n = max_n - min_n if max_n != min_n else 1
    for n in counts:
        ratio = (n - min_n) / range_n
        lw = 3.2 - ratio * 2.2
        line_alpha = 1.0 - ratio * 0.55
        if alpha is not None:
            line_alpha = line_alpha * float(alpha)
        marker_alpha = line_alpha
        marker_size = 18.0 - ratio * 14.0
        show_markers = n <= 200 or max_n <= 50
        if n > 200 and max_n > 50 and ratio > 0.3:
            marker_alpha = line_alpha * 0.5
        weights.append({
            'linewidth': round(lw, 2),
            'alpha': round(line_alpha, 3),
            'marker_alpha': round(marker_alpha, 3),
            'marker_size': round(marker_size, 2),
            'show_markers': show_markers,
        })
    return weights


def plot_ecdf(datasets, title='ECDF 对比图', xlabel='数值', ylabel='累积概率',
              width=800, height=500, dpi=100, alpha=None):
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    font = _get_font_prop()

    counts = []
    computed = []
    for ds in datasets:
        data = ds.get('data', [])
        x, y = compute_ecdf(data)
        computed.append(x)
        counts.append(len(x))
    weights = _calc_visual_weights(counts, alpha=alpha)

    for i, ds in enumerate(datasets):
        name = ds.get('name', f'数据集 {i + 1}')
        x, y = compute_ecdf(ds.get('data', []))
        if len(x) == 0:
            continue
        color = _PLOT_COLORS[i % len(_PLOT_COLORS)]
        w = weights[i] if i < len(weights) else {'linewidth': 2, 'alpha': 1.0,
                                                  'marker_alpha': 1.0, 'marker_size': 12,
                                                  'show_markers': True}
        label_with_n = f'{name}  (n={len(x):,})'
        ax.step(x, y, where='post', label=label_with_n, color=color,
                linewidth=w['linewidth'], alpha=w['alpha'])
        if w['show_markers'] and len(x) <= 1000:
            step_size = max(1, len(x) // 500)
            x_marker = x[::step_size]
            y_marker = y[::step_size]
            ax.scatter(x_marker, y_marker, s=w['marker_size'], color=color,
                       zorder=3, edgecolors='white', linewidths=0.5,
                       alpha=w['marker_alpha'])

    ax.set_title(title, fontproperties=font, fontsize=14)
    ax.set_xlabel(xlabel, fontproperties=font, fontsize=12)
    ax.set_ylabel(ylabel, fontproperties=font, fontsize=12)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(left=0)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(prop=font, loc='lower right', framealpha=0.9)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(font)

    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_b64


def ecdf_statistics(datasets):
    results = []
    for i, ds in enumerate(datasets):
        name = ds.get('name', f'数据集 {i + 1}')
        data = ds.get('data', [])
        arr = np.asarray(data, dtype=float)
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            results.append({
                'name': name, 'count': 0,
                'mean': None, 'std': None,
                'min': None, 'q25': None,
                'median': None, 'q75': None,
                'max': None,
            })
            continue
        results.append({
            'name': name,
            'count': int(len(arr)),
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            'min': float(np.min(arr)),
            'q25': float(np.percentile(arr, 25)),
            'median': float(np.median(arr)),
            'q75': float(np.percentile(arr, 75)),
            'max': float(np.max(arr)),
        })
    return results
