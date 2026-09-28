// 2D panels: fly-eye mosaic, descending-neuron bars, spike raster, perf.

function fitCanvas(canvas) {
  const dpr = Math.min(window.devicePixelRatio, 2);
  const w = Math.max(1, Math.round(canvas.clientWidth * dpr));
  const h = Math.max(1, Math.round(canvas.clientHeight * dpr));
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  return { w, h, dpr };
}

export class EyePanel {
  constructor(canvas, dims) {
    this.canvas = canvas;
    this.az = dims.az;
    this.el = dims.el;
    this.ctx = canvas.getContext('2d');
  }

  draw(eye) {
    if (!eye) return;
    const { w, h } = fitCanvas(this.canvas);
    const g = this.ctx;
    g.clearRect(0, 0, w, h);
    const gap = w * 0.04;
    const eyeW = (w - gap * 3) / 2;
    const r = Math.min(eyeW / (this.az + 0.5) / 1.732, (h - 26) / (this.el * 1.5 + 0.5));
    const hexW = r * 1.732;
    for (let e = 0; e < 2; e++) {
      const ox = gap + e * (eyeW + gap) + (eyeW - hexW * (this.az + 0.5)) / 2 + hexW / 2;
      const oy = (h - r * (this.el * 1.5 + 0.5)) / 2 + r + 6;
      for (let i = 0; i < this.el; i++) {
        for (let j = 0; j < this.az; j++) {
          const v = eye[(e * this.el + i) * this.az + j] / 255;
          const cx = ox + j * hexW + (i % 2 ? hexW / 2 : 0);
          const cy = oy + i * r * 1.5;
          g.beginPath();
          for (let k = 0; k < 6; k++) {
            const a = Math.PI / 6 + (k * Math.PI) / 3;
            const px = cx + r * 0.94 * Math.cos(a), py = cy + r * 0.94 * Math.sin(a);
            k ? g.lineTo(px, py) : g.moveTo(px, py);
          }
          g.closePath();
          const lum = Math.round(20 + v * 235);
          g.fillStyle = `rgb(${Math.round(lum * 1.0)},${Math.round(lum * 0.72)},${Math.round(lum * 0.28)})`;
          g.fill();
        }
      }
      g.fillStyle = 'rgba(200,210,230,0.6)';
      g.font = `${Math.round(10 * Math.min(2, window.devicePixelRatio))}px system-ui`;
      g.textBaseline = 'bottom';
      g.fillText(e ? 'right eye' : 'left eye', ox - hexW / 2, oy - r - 2);
    }
  }
}

export class MotorPanel {
  constructor(el, rows) {
    this.rows = rows;
    this.el = el;
    el.innerHTML = '';
    this.bars = {};
    for (const row of rows) {
      const div = document.createElement('div');
      div.className = 'mrow';
      div.style.setProperty('--c', row.color);
      const split = row.kind === 'split';
      div.innerHTML = `
        <div class="name">${row.label} <small>${row.neurons}</small></div>
        <div class="bar ${split ? 'split' : ''}">${split ? '<b class="l"></b><b class="r"></b>' : '<b></b>'}</div>
        <div class="val">0</div>`;
      el.appendChild(div);
      this.bars[row.id] = {
        row,
        l: div.querySelector(split ? 'b.l' : 'b'),
        r: split ? div.querySelector('b.r') : null,
        val: div.querySelector('.val'),
      };
    }
  }

  update(motor) {
    for (const id in this.bars) {
      const b = this.bars[id];
      const v = motor[id];
      if (v === undefined) continue;
      const max = b.row.maxHz;
      if (b.r) {
        b.l.style.width = `${Math.min(50, (v[0] / max) * 50)}%`;
        b.r.style.width = `${Math.min(50, (v[1] / max) * 50)}%`;
        b.val.textContent = `${Math.round(v[0])}|${Math.round(v[1])}`;
      } else {
        b.l.style.width = `${Math.min(100, (v / max) * 100)}%`;
        b.val.textContent = `${Math.round(v)}`;
      }
    }
  }
}

export class RasterPanel {
  constructor(canvas, keyRows, classes) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.keyRows = keyRows; // [{label, color}] one per key neuron
    this.classes = classes;
    this.history = [];
    this.maxCols = 400;
  }

  push(keyCounts, classCounts) {
    this.history.push({ key: keyCounts, cls: classCounts });
    if (this.history.length > this.maxCols) this.history.shift();
  }

  draw() {
    const { w, h, dpr } = fitCanvas(this.canvas);
    const g = this.ctx;
    g.clearRect(0, 0, w, h);
    const labelW = 64 * dpr;
    const rasterH = h * 0.62;
    const rows = this.keyRows.length || 1;
    const rh = rasterH / rows;
    const cols = this.history.length;
    const cw = (w - labelW) / this.maxCols;
    const x0 = w - cols * cw;

    // Group labels (one per run of same-coloured rows).
    g.font = `${Math.round(9 * dpr)}px ui-monospace, monospace`;
    g.textBaseline = 'middle';
    let start = 0;
    for (let i = 1; i <= rows; i++) {
      if (i === rows || this.keyRows[i].group !== this.keyRows[start].group) {
        const y = ((start + i) / 2) * rh;
        g.fillStyle = this.keyRows[start].color;
        g.fillText(this.keyRows[start].group, 4 * dpr, y);
        g.fillStyle = 'rgba(255,255,255,0.05)';
        g.fillRect(labelW, i * rh - 0.5, w - labelW, 1);
        start = i;
      }
    }
    for (let c = 0; c < cols; c++) {
      const col = this.history[c].key;
      const x = x0 + c * cw;
      for (let r = 0; r < rows; r++) {
        const v = col[r];
        if (!v) continue;
        g.fillStyle = this.keyRows[r].color;
        g.globalAlpha = Math.min(1, 0.45 + 0.3 * v);
        g.fillRect(x, r * rh + rh * 0.12, Math.max(1, cw * 0.9), rh * 0.76);
      }
    }
    g.globalAlpha = 1;

    // Whole-CNS activity, stacked by class.
    const y0 = rasterH + 6 * dpr;
    const ah = h - y0 - 2;
    let peak = 1;
    for (const col of this.history) { let s = 0; for (const v of col.cls) s += v; peak = Math.max(peak, s); }
    this.peak = Math.max(peak, (this.peak || 1) * 0.995);
    for (let c = 0; c < cols; c++) {
      const col = this.history[c].cls;
      const x = x0 + c * cw;
      let y = y0 + ah;
      for (let k = 0; k < col.length; k++) {
        const hh = (col[k] / this.peak) * ah;
        if (hh <= 0) continue;
        g.fillStyle = this.classes[k]?.color || '#888';
        g.fillRect(x, y - hh, Math.max(1, cw), hh);
        y -= hh;
      }
    }
    g.fillStyle = 'rgba(200,210,230,0.55)';
    g.fillText(`CNS spikes/frame (peak ${Math.round(this.peak)})`, 4 * dpr, y0 + 6 * dpr);
  }
}

export class PerfPanel {
  constructor(gridEl, canvas) {
    this.grid = gridEl;
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.fields = {};
    const spec = [
      ['fps', 'render fps'], ['frame', 'frame ms'], ['points', 'points drawn'],
      ['calls', 'draw calls'], ['upload', 'GPU upload/s'], ['ws', 'socket in'],
      ['rtf', 'sim speed'], ['gpu', 'brain ms/frame'], ['spk', 'spikes/s'],
    ];
    gridEl.innerHTML = spec.map(([k, l]) => `<div><span>${l}</span><b id="perf-${k}">–</b></div>`).join('');
    for (const [k] of spec) this.fields[k] = document.getElementById(`perf-${k}`);
    this.fpsHist = [];
  }

  set(k, v) { if (this.fields[k]) this.fields[k].textContent = v; }

  pushFps(fps) {
    this.fpsHist.push(fps);
    if (this.fpsHist.length > 120) this.fpsHist.shift();
    const { w, h, dpr } = fitCanvas(this.canvas);
    const g = this.ctx;
    g.clearRect(0, 0, w, h);
    const max = Math.max(144, ...this.fpsHist);
    g.strokeStyle = 'rgba(255,255,255,0.12)';
    for (const ref of [30, 60, 120]) {
      const y = h - (ref / max) * h;
      g.beginPath(); g.moveTo(0, y); g.lineTo(w, y); g.stroke();
      g.fillStyle = 'rgba(255,255,255,0.3)';
      g.font = `${Math.round(9 * dpr)}px ui-monospace, monospace`;
      g.fillText(`${ref}`, 2, y - 2);
    }
    g.strokeStyle = '#7cf3ff';
    g.lineWidth = 1.5 * dpr;
    g.beginPath();
    this.fpsHist.forEach((f, i) => {
      const x = (i / 119) * w, y = h - (f / max) * h;
      i ? g.lineTo(x, y) : g.moveTo(x, y);
    });
    g.stroke();
  }
}
