import json
import os
import tempfile
from flask import Flask, request, jsonify, render_template_string, send_from_directory
from ecdf_utils import compute_ecdf, plot_ecdf, ecdf_statistics

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


HTML_PAGE = r"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ECDF 经验累积分布函数图服务</title>
<style>
  :root {
    --bg: #f4f6f9; --card: #ffffff; --border: #dde1e6;
    --primary: #1f77b4; --primary-light: #4a9fe5;
    --text: #1a1a2e; --muted: #6b7280;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, 'Segoe UI', 'Microsoft YaHei', sans-serif;
         background: var(--bg); color: var(--text); min-height: 100vh; }
  .header { background: linear-gradient(135deg, var(--primary), var(--primary-light));
             color: #fff; padding: 28px 0; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,.15); }
  .header h1 { font-size: 26px; font-weight: 700; }
  .header p { margin-top: 6px; opacity: .88; font-size: 14px; }
  .container { max-width: 960px; margin: 32px auto; padding: 0 20px; }
  .card { background: var(--card); border-radius: 12px; padding: 24px;
          box-shadow: 0 1px 4px rgba(0,0,0,.06); margin-bottom: 24px; border: 1px solid var(--border); }
  .card h2 { font-size: 17px; margin-bottom: 16px; display: flex; align-items: center; gap: 8px; }
  .card h2::before { content: ''; display: inline-block; width: 4px; height: 18px;
                      background: var(--primary); border-radius: 2px; }
  .dataset-row { display: flex; gap: 12px; margin-bottom: 12px; align-items: flex-start; flex-wrap: wrap; }
  .dataset-row input[type="text"] { flex: 0 0 160px; padding: 8px 12px; border: 1px solid var(--border);
                                     border-radius: 6px; font-size: 14px; }
  .dataset-row textarea { flex: 1; min-width: 240px; min-height: 68px; padding: 8px 12px;
                           border: 1px solid var(--border); border-radius: 6px; font-size: 13px;
                           font-family: 'Consolas', 'Courier New', monospace; resize: vertical; }
  .dataset-row .del-btn { padding: 8px 12px; background: #fee2e2; color: #dc2626;
                           border: none; border-radius: 6px; cursor: pointer; font-size: 13px; }
  .dataset-row .del-btn:hover { background: #fecaca; }
  .actions { display: flex; gap: 12px; margin-top: 16px; flex-wrap: wrap; }
  .btn { padding: 10px 24px; border: none; border-radius: 8px; cursor: pointer;
         font-size: 14px; font-weight: 600; transition: all .2s; }
  .btn-primary { background: var(--primary); color: #fff; }
  .btn-primary:hover { background: var(--primary-light); }
  .btn-outline { background: #fff; color: var(--primary); border: 1.5px solid var(--primary); }
  .btn-outline:hover { background: #eef6fd; }
  .btn-sm { padding: 6px 14px; font-size: 13px; }
  .settings { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
              gap: 12px; margin-top: 12px; }
  .settings label { display: flex; flex-direction: column; gap: 4px; font-size: 13px; color: var(--muted); }
  .settings input { padding: 8px 12px; border: 1px solid var(--border); border-radius: 6px; font-size: 14px; }
  #result { text-align: center; }
  #result img { max-width: 100%; border-radius: 8px; box-shadow: 0 2px 12px rgba(0,0,0,.1); }
  .stats-table { width: 100%; border-collapse: collapse; margin-top: 16px; font-size: 13px; }
  .stats-table th, .stats-table td { padding: 8px 10px; border-bottom: 1px solid var(--border); text-align: right; }
  .stats-table th { background: #f8f9fb; color: var(--muted); font-weight: 600; text-align: right; }
  .stats-table th:first-child, .stats-table td:first-child { text-align: left; }
  .upload-hint { font-size: 12px; color: var(--muted); margin-top: 4px; }
  .divider { border: none; border-top: 1px dashed var(--border); margin: 12px 0; }
  .empty-msg { color: var(--muted); font-size: 14px; text-align: center; padding: 32px 0; }
</style>
</head>
<body>
<div class="header">
  <h1>📊 ECDF 经验累积分布函数图</h1>
  <p>上传多组数据，生成 ECDF 曲线对比图 — 无需假设分布形态</p>
</div>
<div class="container">
  <div class="card">
    <h2>数据输入</h2>
    <div id="datasets"></div>
    <div class="actions">
      <button class="btn btn-outline btn-sm" onclick="addDataset()">＋ 添加数据集</button>
      <button class="btn btn-outline btn-sm" onclick="loadSample()">📋 加载示例数据</button>
      <label class="btn btn-outline btn-sm" style="display:inline-flex;align-items:center;">
        📁 上传 CSV
        <input type="file" accept=".csv,.txt" multiple style="display:none" onchange="handleFiles(event)">
      </label>
    </div>
    <hr class="divider">
    <h2>图表设置</h2>
    <div class="settings">
      <label>图表标题 <input id="chartTitle" value="ECDF 对比图"></label>
      <label>X 轴标签 <input id="xLabel" value="数值"></label>
      <label>Y 轴标签 <input id="yLabel" value="累积概率"></label>
      <label>图片宽度 <input id="imgWidth" type="number" value="800" min="400" max="1600"></label>
      <label>图片高度 <input id="imgHeight" type="number" value="500" min="300" max="1000"></label>
    </div>
    <div class="actions" style="margin-top:20px;">
      <button class="btn btn-primary" onclick="generate()">🎨 生成 ECDF 图</button>
      <button class="btn btn-outline" onclick="downloadImg()">💾 下载图片</button>
    </div>
  </div>
  <div class="card" id="resultCard" style="display:none;">
    <h2>分析结果</h2>
    <div id="stats"></div>
    <hr class="divider">
    <div id="result"></div>
  </div>
</div>

<script>
let datasets = [];
let dsCounter = 0;
let lastImgB64 = '';

function addDataset(name, values) {
  const id = ++dsCounter;
  datasets.push({ id, name: name || ('数据集 ' + id), values: values || '' });
  render();
}

function removeDataset(id) {
  datasets = datasets.filter(d => d.id !== id);
  render();
}

function render() {
  const box = document.getElementById('datasets');
  if (datasets.length === 0) {
    box.innerHTML = '<div class="empty-msg">点击「添加数据集」开始输入数据，或加载示例数据</div>';
    return;
  }
  box.innerHTML = datasets.map(d => `
    <div class="dataset-row">
      <input type="text" value="${escHtml(d.name)}" placeholder="数据集名称"
             onchange="datasets.find(x=>x.id===${d.id}).name=this.value">
      <textarea placeholder="输入数值，用逗号、空格或换行分隔&#10;例如: 1.2, 3.4, 5.6, 7.8"
                onchange="datasets.find(x=>x.id===${d.id}).values=this.value">${escHtml(d.values)}</textarea>
      <button class="del-btn" onclick="removeDataset(${d.id})">删除</button>
    </div>
  `).join('');
}

function escHtml(s) { return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/"/g,'&quot;'); }

function parseValues(raw) {
  return raw.split(/[,\s\n\r\t]+/).map(s => s.trim()).filter(s => s !== '').map(Number).filter(n => !isNaN(n));
}

function loadSample() {
  datasets = [];
  dsCounter = 0;
  const n = 80;
  const a = [], b = [], c = [];
  for (let i = 0; i < n; i++) {
    a.push(+(Math.random() * 40 + 10).toFixed(1));
    b.push(+(Math.random() * 30 + 30).toFixed(1));
    c.push(+(Math.random() * 50 + 5).toFixed(1));
  }
  addDataset('均匀分布 A (10~50)', a.join(', '));
  addDataset('均匀分布 B (30~60)', b.join(', '));
  addDataset('均匀分布 C (5~55)', c.join(', '));
}

function handleFiles(event) {
  const files = event.target.files;
  Array.from(files).forEach(file => {
    const reader = new FileReader();
    reader.onload = e => {
      const text = e.target.result;
      const lines = text.trim().split(/\r?\n/);
      const name = file.name.replace(/\.\w+$/, '');
      let values;
      if (lines.length === 1 || (lines.length > 1 && lines[0].includes(','))) {
        values = lines.join(', ');
      } else {
        values = lines.join(', ');
      }
      addDataset(name, values);
    };
    reader.readAsText(file);
  });
  event.target.value = '';
}

async function generate() {
  if (datasets.length === 0) { alert('请先添加至少一组数据集'); return; }
  const payload = {
    datasets: datasets.map(d => ({ name: d.name, data: parseValues(d.values) })),
    title: document.getElementById('chartTitle').value,
    xlabel: document.getElementById('xLabel').value,
    ylabel: document.getElementById('yLabel').value,
    width: parseInt(document.getElementById('imgWidth').value) || 800,
    height: parseInt(document.getElementById('imgHeight').value) || 500,
  };
  const resp = await fetch('/api/ecdf', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!resp.ok) { const err = await resp.json(); alert('错误: ' + (err.error || '未知错误')); return; }
  const result = await resp.json();
  lastImgB64 = result.image;
  document.getElementById('result').innerHTML = `<img src="data:image/png;base64,${result.image}" alt="ECDF 图">`;
  document.getElementById('stats').innerHTML = buildStatsTable(result.statistics);
  document.getElementById('resultCard').style.display = '';
}

function buildStatsTable(stats) {
  if (!stats || stats.length === 0) return '';
  const cols = ['name','count','mean','std','min','q25','median','q75','max'];
  const heads = ['数据集','样本量','均值','标准差','最小值','Q25','中位数','Q75','最大值'];
  let h = '<table class="stats-table"><thead><tr>' + heads.map(t=>`<th>${t}</th>`).join('') + '</tr></thead><tbody>';
  stats.forEach(s => {
    h += '<tr>' + cols.map(c => {
      const v = s[c];
      return c === 'name' ? `<td>${escHtml(v)}</td>` : `<td>${v != null ? (typeof v === 'number' ? v.toFixed(2) : v) : '-'}</td>`;
    }).join('') + '</tr>';
  });
  h += '</tbody></table>';
  return h;
}

function downloadImg() {
  if (!lastImgB64) { alert('请先生成图表'); return; }
  const a = document.createElement('a');
  a.href = 'data:image/png;base64,' + lastImgB64;
  a.download = 'ecdf_chart.png';
  a.click();
}

render();
</script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(HTML_PAGE)


@app.route('/api/ecdf', methods=['POST'])
def api_ecdf():
    body = request.get_json(force=True)
    datasets = body.get('datasets', [])
    if not datasets:
        return jsonify({'error': '至少需要一组数据'}), 400

    parsed = []
    for ds in datasets:
        name = ds.get('name', '未命名')
        data = ds.get('data', [])
        if not data:
            return jsonify({'error': f'数据集「{name}」没有有效数据'}), 400
        parsed.append({'name': name, 'data': data})

    title = body.get('title', 'ECDF 对比图')
    xlabel = body.get('xlabel', '数值')
    ylabel = body.get('ylabel', '累积概率')
    width = body.get('width', 800)
    height = body.get('height', 500)

    img_b64 = plot_ecdf(parsed, title=title, xlabel=xlabel, ylabel=ylabel,
                         width=width, height=height)
    stats = ecdf_statistics(parsed)

    return jsonify({'image': img_b64, 'statistics': stats})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
