// Display timing only: never predicts physics or creates spike events.
export const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

export class ArrivalCadence {
  constructor() { this.reset(); }
  reset() { this.last = null; this.interval = 0.15; }
  observe(now) {
    if (this.last !== null) {
      const gap = clamp(now - this.last, 1 / 120, 2);
      this.interval += (gap - this.interval) * (gap > this.interval ? 0.6 : 0.25);
    }
    this.last = now;
    return clamp(this.interval, 1 / 60, 1);
  }
}

export function progress(now, start, duration) {
  return duration > 0 ? clamp((now - start) / duration, 0, 1) : 1;
}

export function interpolatePose(from, to, amount) {
  const result = {};
  for (const key of ['x', 'y', 'z', 't']) {
    result[key] = (from[key] || 0) + ((to[key] || 0) - (from[key] || 0)) * amount;
  }
  const dh = Math.atan2(Math.sin(to.h - from.h), Math.cos(to.h - from.h));
  result.h = from.h + dh * amount;
  return result;
}

// The protocol supplies unique firing neurons, not per-neuron spike counts.
// Smooth participation in recent simulation windows, without calling it Hz.
export class RecentActivity {
  constructor(n) {
    this.from = new Float32Array(n);
    this.target = new Float32Array(n);
    this.lastSpike = new Float32Array(n);
    this.reset();
  }
  reset() {
    this.from.fill(0);
    this.target.fill(0);
    this.lastSpike.fill(-1e4);
    this.start = 0;
    this.duration = 0.15;
  }
  update(indices, now, simSeconds, duration) {
    const p = progress(now, this.start, this.duration);
    // After the expected next packet, stale activity fades instead of staying lit.
    const fade = Math.exp(-Math.max(0, now - this.start - this.duration) / (2 * this.duration));
    const alpha = 1 - Math.exp(-Math.max(0, simSeconds) / 0.07);
    for (let i = 0; i < this.target.length; i++) {
      this.from[i] = (this.from[i] + (this.target[i] - this.from[i]) * p) * fade;
      this.target[i] *= (1 - alpha) * fade;
    }
    for (const i of indices) {
      this.target[i] += alpha;
      this.lastSpike[i] = now;
    }
    this.start = now;
    this.duration = Math.max(1 / 60, duration);
  }
}
