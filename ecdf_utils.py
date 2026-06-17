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


def _ks_p_value(d, n1, n2):
    if d <= 0:
        return 1.0
    en = np.sqrt(n1 * n2 / (n1 + n2))
    lam = (en + 0.12 + 0.11 / en) * d
    if lam < 0.27:
        return 1.0
    if lam >= 7.5:
        return 0.0
    q = 0.0
    for k in range(1, 201):
        term = 2.0 * ((-1.0) ** (k - 1)) * np.exp(-2.0 * k * k * lam * lam)
        q += term
        if abs(term) < 1e-8:
            break
    return max(0.0, min(1.0, q))


def ks_two_sample(data1, data2):
    arr1 = np.asarray(data1, dtype=float)
    arr2 = np.asarray(data2, dtype=float)
    arr1 = arr1[~np.isnan(arr1)]
    arr2 = arr2[~np.isnan(arr2)]
    n1, n2 = len(arr1), len(arr2)
    if n1 == 0 or n2 == 0:
        return {'statistic': None, 'p_value': None, 'n1': n1, 'n2': n2, 'x_at_max': None}
    x1 = np.sort(arr1)
    x2 = np.sort(arr2)
    all_x = np.unique(np.concatenate([x1, x2]))
    ecdf1 = np.searchsorted(x1, all_x, side='right') / n1
    ecdf2 = np.searchsorted(x2, all_x, side='right') / n2
    diff = np.abs(ecdf1 - ecdf2)
    idx = np.argmax(diff)
    d = float(diff[idx])
    x_max = float(all_x[idx])
    p = _ks_p_value(d, n1, n2)
    return {'statistic': d, 'p_value': p, 'n1': n1, 'n2': n2, 'x_at_max': x_max}


def pairwise_ks_tests(datasets):
    results = []
    cleaned = []
    for ds in datasets:
        name = ds.get('name', '未命名')
        arr = np.asarray(ds.get('data', []), dtype=float)
        arr = arr[~np.isnan(arr)]
        cleaned.append({'name': name, 'data': arr})
    m = len(cleaned)
    for i in range(m):
        for j in range(i + 1, m):
            res = ks_two_sample(cleaned[i]['data'], cleaned[j]['data'])
            results.append({
                'i': i, 'j': j,
                'name_i': cleaned[i]['name'],
                'name_j': cleaned[j]['name'],
                'statistic': res['statistic'],
                'p_value': res['p_value'],
                'x_at_max': res['x_at_max'],
            })
    return results


def plot_ecdf(datasets, title='ECDF 对比图', xlabel='数值', ylabel='累积概率',
              width=800, height=500, dpi=100, alpha=None, show_ks=True):
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    font = _get_font_prop()

    counts = []
    ecdf_data = []
    names = []
    for ds in datasets:
        name = ds.get('name', f'数据集 {len(names) + 1}')
        data = ds.get('data', [])
        x, y = compute_ecdf(data)
        ecdf_data.append((x, y))
        counts.append(len(x))
        names.append(name)
    weights = _calc_visual_weights(counts, alpha=alpha)

    for i in range(len(ecdf_data)):
        x, y = ecdf_data[i]
        if len(x) == 0:
            continue
        color = _PLOT_COLORS[i % len(_PLOT_COLORS)]
        w = weights[i] if i < len(weights) else {'linewidth': 2, 'alpha': 1.0,
                                                  'marker_alpha': 1.0, 'marker_size': 12,
                                                  'show_markers': True}
        label_with_n = f'{names[i]}  (n={len(x):,})'
        ax.step(x, y, where='post', label=label_with_n, color=color,
                linewidth=w['linewidth'], alpha=w['alpha'])
        if w['show_markers'] and len(x) <= 1000:
            step_size = max(1, len(x) // 500)
            x_marker = x[::step_size]
            y_marker = y[::step_size]
            ax.scatter(x_marker, y_marker, s=w['marker_size'], color=color,
                       zorder=3, edgecolors='white', linewidths=0.5,
                       alpha=w['marker_alpha'])

    ks_results = []
    if show_ks and len(datasets) >= 2:
        ks_results = pairwise_ks_tests(datasets)
        valid_ks = [r for r in ks_results if r['statistic'] is not None]
        if valid_ks:
            text_lines = ['KS 检验 (两两对比)']
            for r in valid_ks:
                d_val = r['statistic']
                p_val = r['p_value']
                if p_val < 0.001:
                    p_str = 'p < 0.001'
                else:
                    p_str = f'p = {p_val:.3f}'
                sig = ' ***' if p_val < 0.001 else (' **' if p_val < 0.01 else (' *' if p_val < 0.05 else ''))
                text_lines.append(f"{r['name_i']} vs {r['name_j']}:  D={d_val:.3f}, {p_str}{sig}")
            text_str = '\n'.join(text_lines)
            ax.text(0.02, 0.98, text_str, transform=ax.transAxes,
                    fontsize=9, va='top', ha='left',
                    fontproperties=font,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                              edgecolor='#cccccc', alpha=0.92))

            for r in valid_ks:
                if r['x_at_max'] is None:
                    continue
                i_idx, j_idx = r['i'], r['j']
                xi, yi = ecdf_data[i_idx]
                xj, yj = ecdf_data[j_idx]
                if len(xi) == 0 or len(xj) == 0:
                    continue
                ecdf_i = np.searchsorted(xi, [r['x_at_max']], side='right')[0] / len(xi)
                ecdf_j = np.searchsorted(xj, [r['x_at_max']], side='right')[0] / len(xj)
                y_lo = min(ecdf_i, ecdf_j)
                y_hi = max(ecdf_i, ecdf_j)
                color_i = _PLOT_COLORS[i_idx % len(_PLOT_COLORS)]
                ax.annotate('', xy=(r['x_at_max'], y_hi), xytext=(r['x_at_max'], y_lo),
                            arrowprops=dict(arrowstyle='<->', color='red', lw=1.3, alpha=0.8))
                if r['p_value'] < 0.05:
                    mid_y = (y_lo + y_hi) / 2
                    ax.text(r['x_at_max'], mid_y, f' D={r["statistic"]:.2f}',
                            fontsize=8, va='center', ha='left',
                            color='darkred', fontproperties=font,
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='#fff0f0',
                                      edgecolor='red', alpha=0.85))

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
