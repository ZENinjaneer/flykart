import test from 'node:test';
import assert from 'node:assert/strict';
import { ArrivalCadence, RecentActivity, interpolatePose, progress } from '../js/presentation.js';
import { EyePanel, RasterPanel } from '../js/dashboard.js';

test('cadence adapts to slow packets and does not absorb a pause after reset', () => {
  const c = new ArrivalCadence();
  c.observe(0);
  c.observe(0.6);
  assert.ok(c.observe(1.2) > 0.5);
  c.reset();
  assert.equal(c.observe(100), 0.15);
  assert.ok(c.observe(100.03) < 0.15);
});

test('interpolation reaches the received position and never extrapolates during stalls', () => {
  const from = { x: 0, y: 1, z: 0, h: Math.PI - 0.1, t: 0 };
  const to = { x: 6, y: 1, z: 2, h: -Math.PI + 0.1, t: 0.0144 };
  const midway = interpolatePose(from, to, progress(10.3, 10, 0.6));
  assert.ok(Math.abs(midway.x - 3) < 1e-10);
  assert.ok(Math.abs(midway.h - Math.PI) < 1e-10);
  const stalled = interpolatePose(from, to, progress(99, 10, 0.6));
  assert.equal(stalled.x, to.x);
  assert.equal(stalled.t, to.t);
  assert.equal(progress(9, 10, 0.6), 0);
  assert.equal(progress(10, 10, 0), 1);
});

test('recent activity uses only received spikes, smooths the transition, and resets', () => {
  const a = new RecentActivity(3);
  a.update([1], 1, 0.0144, 0.6);
  assert.equal(a.from[1], 0);
  assert.ok(a.target[1] > 0 && a.target[1] < 1);
  assert.equal(a.lastSpike[1], 1);
  const first = a.target[1];
  a.update([1], 1.6, 0.0144, 0.6);
  assert.ok(Math.abs(a.from[1] - first) < 1e-6);
  assert.ok(a.target[1] > first);
  assert.equal(a.target[0], 0);
  assert.equal(a.target[2], 0);
  a.update([], 99, 0.0144, 0.6);
  assert.ok(a.target[1] < 1e-20);
  assert.equal(a.lastSpike[1], Math.fround(1.6));
  a.reset();
  assert.deepEqual([...a.target], [0, 0, 0]);
  assert.ok(a.lastSpike.every(v => v < 0));
});

function canvasFixture() {
  let clears = 0;
  const ctx = new Proxy({}, { get: (_, key) => key === 'clearRect' ? () => clears++ : () => {} });
  return {
    canvas: { width: 0, height: 0, clientWidth: 200, clientHeight: 100, getContext: () => ctx },
    get clears() { return clears; },
  };
}
globalThis.window = { devicePixelRatio: 1 };
globalThis.ResizeObserver = class {
  constructor(callback) { this.callback = callback; }
  observe(canvas) { canvas.resize = this.callback; }
};

test('eye redraws only for changed data or size', () => {
  const f = canvasFixture();
  const p = new EyePanel(f.canvas, { az: 1, el: 1 });
  p.draw([0, 255]);
  assert.equal(f.clears, 1);
  p.draw([0, 255]);
  assert.equal(f.clears, 1);
  p.draw([255, 255]);
  assert.equal(f.clears, 2);
  f.canvas.clientWidth = 300;
  f.canvas.resize();
  p.draw([255, 255]);
  assert.equal(f.clears, 3);
  assert.equal(f.canvas.width, 300);
});

test('raster redraws only for new simulation windows, resize or reset', () => {
  const f = canvasFixture();
  const p = new RasterPanel(f.canvas, [{ group: 'key', color: '#fff' }], [{ color: '#fff' }]);
  p.draw();
  p.draw();
  assert.equal(f.clears, 1);
  p.push([1], [5]);
  p.draw();
  p.draw();
  assert.equal(f.clears, 2);
  f.canvas.resize();
  p.draw();
  assert.equal(f.clears, 3);
  p.reset();
  p.draw();
  assert.equal(p.history.length, 0);
  assert.equal(f.clears, 4);
});
