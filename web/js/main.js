import { BrainView } from './brainView.js';
import { KartView } from './kartView.js';
import { EyePanel, MotorPanel, RasterPanel, PerfPanel } from './dashboard.js';
import { loadModels } from './models.js';
import { ArrivalCadence } from './presentation.js';

const $ = (id) => document.getElementById(id);
const decoder = new TextDecoder();

let neurons = null;
let init = null;
let brain = null;
let kart = null;
let eyePanel = null;
let motorPanel = null;
let raster = null;
let perf = null;
let socket = null;
let running = false;
let lastEvents = null;
let lastEye = null;
let flashTimer = 0;
let simTime = 0;
let simRate = 0;
let connected = false;
let epoch = null;
let lastBrainTime = null;
const cadence = new ArrivalCadence();

const net = { msgs: 0, bytes: 0 };
const frameStats = { n: 0, cpu: 0, upload: 0, t0: performance.now() };

async function loadNeurons() {
  const meta = await (await fetch('/data/neurons.json')).json();
  const buf = await (await fetch('/data/neurons.bin')).arrayBuffer();
  const L = meta.layout;
  return {
    n: meta.n,
    meta,
    pos: new Float32Array(buf, L.pos, meta.n * 3),
    type: new Uint16Array(buf, L.type, meta.n),
    cls: new Uint8Array(buf, L.cls, meta.n),
    side: new Uint8Array(buf, L.side, meta.n),
  };
}

function send(obj) {
  if (socket && socket.readyState === 1) socket.send(JSON.stringify(obj));
}

function setConn(on) {
  connected = on;
  const el = $('conn');
  el.textContent = on ? 'live' : 'offline';
  el.className = `conn ${on ? 'on' : 'off'}`;
  $('btn-run').disabled = !on;
  $('btn-reset').disabled = !on;
  if (!on) {
    setRunning(false);
    cadence.reset();
  }
  updateKartStatus();
}

function connect() {
  const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
  ws.binaryType = 'arraybuffer';
  ws.onopen = () => setConn(true);
  ws.onclose = () => { setConn(false); setTimeout(connect, 1000); };
  ws.onmessage = (ev) => {
    net.msgs++;
    if (typeof ev.data === 'string') {
      net.bytes += ev.data.length;
      onText(JSON.parse(ev.data));
    } else {
      net.bytes += ev.data.byteLength;
      onFrame(ev.data);
    }
  };
  socket = ws;
}

function onText(msg) {
  if (msg.type === 'init') onInit(msg);
  else if (msg.type === 'status') onStatus(msg);
}

function onInit(msg) {
  const first = !init;
  init = msg;
  if (first) {
    kart = new KartView($('kart-canvas'), $('kart-canvas').parentElement, msg.world);
    loadModels().then((models) => {
      if (!models.length) return;
      const credits = kart.applyModels(models);
      if (credits.length) $('kart-credits').textContent = `3D art: ${credits.join(' · ')}`;
    });
    eyePanel = new EyePanel($('eye-canvas'), msg.world.eye);
    motorPanel = new MotorPanel($('motor-rows'), msg.motorRows);
    raster = new RasterPanel($('raster-canvas'), msg.keyRows, neurons.meta.classes);
    brain.setGroups(msg.groups, $('labels'));
    buildPokes(msg.pokes);
    requestAnimationFrame(loop);
  }
  clearPresentation();
  epoch = null;
  const m = msg.meta;
  $('subtitle').textContent =
    `${m.dataset} · ${m.neurons.toLocaleString()} neurons · ${(m.connections / 1e6).toFixed(1)}M connections · ` +
    `${(m.synapses / 1e6).toFixed(0)}M synapses · ${m.device}`;
  $('subtitle').title = $('subtitle').textContent;
  onStatus(msg.status);
}

function onStatus(st) {
  if (!acceptEpoch(st.epoch)) return;
  setRunning(st.running);
  $('sel-mode').value = st.mode;
  $('rng-drive').value = Math.round(st.settings.drive * 100);
  $('out-drive').textContent = Math.round(st.settings.drive * 100);
  $('rng-assist').value = Math.round(st.settings.assist * 100);
  $('out-assist').textContent = `${Math.round(st.settings.assist * 100)}%`;
  $('sel-speed').value = String(st.settings.speed);
  updateKartStatus();
}

function setRunning(on) {
  if (running !== on) cadence.reset();
  running = on;
  kart?.setRunning(on);
  $('btn-run').textContent = running ? '❚❚ Pause' : '▶ Run';
}

function clearPresentation() {
  brain.resetActivity();
  kart?.reset();
  raster?.reset();
  cadence.reset();
  lastBrainTime = null;
  lastEvents = null;
  lastEye = null;
  simTime = 0;
  simRate = 0;
  clearTimeout(flashTimer);
  $('hud-flash').classList.remove('show');
}

function acceptEpoch(next) {
  if (next === undefined) return true;
  if (epoch !== null && next < epoch) return false;
  if (epoch !== next) clearPresentation();
  epoch = next;
  return true;
}

function updateKartStatus() {
  if (!connected) {
    $('kart-status').textContent = `Offline · ${simTime.toFixed(2)} s elapsed · reconnecting…`;
    return;
  }
  $('kart-status').textContent = running
    ? `Running · ${simTime.toFixed(2)} s elapsed · ${simRate.toFixed(2)}× real time`
    : `Paused · ${simTime.toFixed(2)} s elapsed · press Run to drive`;
}

function onFrame(buf) {
  const dv = new DataView(buf);
  const jlen = dv.getUint32(0, true);
  const f = JSON.parse(decoder.decode(new Uint8Array(buf, 4, jlen)));
  if (!acceptEpoch(f.epoch)) return;
  if (f.world.t < simTime) clearPresentation();
  if (typeof f.running === 'boolean') setRunning(f.running);
  simTime = f.world.t;
  simRate = f.perf.rtf;
  updateKartStatus();
  let off = (4 + jlen + 3) & ~3;
  const idx = new Uint32Array(buf, off, f.nSpk);
  const now = performance.now() / 1000;
  const advancedMs = f.frameMs ?? (lastBrainTime === null
    ? (f.tMs > 0 ? init.meta.frameMs : 0) : Math.max(0, f.tMs - lastBrainTime));
  lastBrainTime = f.tMs;
  const duration = advancedMs > 0 ? cadence.observe(now) : cadence.interval;
  if (advancedMs > 0) brain.addSpikes(idx, now, advancedMs / 1000, duration);

  kart.update(f.world, now, duration);
  motorPanel.update(f.motor);
  if (advancedMs > 0) raster.push(f.keySpikes, f.classCounts);
  lastEye = f.eye || lastEye;
  updateSenses(f.senses);
  updateHud(f.world, now);
  perf.set('rtf', `${f.perf.rtf.toFixed(2)}×`);
  perf.set('time', `${f.world.t.toFixed(2)} s`);
  perf.set('gpu', f.perf.brainMs.toFixed(1));
  perf.set('spk', Math.round(f.perf.spikesPerSec).toLocaleString());
}

function bar(id, v) {
  const el = $(id);
  if (el) el.style.setProperty('--v', `${Math.round(Math.min(1, v) * 100)}%`);
}

function updateSenses(s) {
  bar('ch-target-l', s.target_l); bar('ch-target-r', s.target_r);
  bar('ch-loom-l', s.loom_l); bar('ch-loom-r', s.loom_r);
  bar('ch-sugar', s.sugar);
}

function flash(text, color) {
  const el = $('hud-flash');
  el.textContent = text;
  el.style.textShadow = `0 0 18px ${color}, 0 4px 18px rgba(0,0,0,0.6)`;
  el.classList.add('show');
  clearTimeout(flashTimer);
  flashTimer = setTimeout(() => el.classList.remove('show'), 700);
}

function updateHud(w, now) {
  const k = w.kart;
  $('hud-speed').textContent = Math.round(Math.abs(k.v) * 3.6 * 2);
  $('hud-wheel').style.transform = `rotate(${k.steer * 120}deg)`;
  const e = w.events;
  $('hud-sugar').textContent = e.sugar;
  $('hud-cleared').textContent = e.cleared;
  $('hud-crash').textContent = e.crashes;
  $('hud-laps').textContent = e.laps.toFixed(2);
  if (lastEvents) {
    if (e.cleared > lastEvents.cleared) flash('GIANT FIBER JUMP!', '#7dff9b');
    else if (e.jumps > lastEvents.jumps) flash('JUMP!', '#7cf3ff');
    if (e.crashes > lastEvents.crashes) flash('SPLAT!', '#ff5b6e');
    if (e.sugar > lastEvents.sugar) flash('SLURP! 🍬', '#ffcf5a');
  }
  if (k.rev && !(lastEvents && lastEvents.rev)) flash('MOONWALK ◀◀', '#ff9f5b');
  lastEvents = { ...e, rev: k.rev };
}

function buildPokes(pokes) {
  const el = $('poke-buttons');
  el.innerHTML = '';
  for (const p of pokes) {
    const b = document.createElement('button');
    b.innerHTML = `${p.label}<small>${p.sub}</small>`;
    b.title = p.title || '';
    b.onclick = () => {
      send({ cmd: 'poke', id: p.id });
      b.classList.add('fired');
      setTimeout(() => b.classList.remove('fired'), 400);
    };
    el.appendChild(b);
  }
}

function buildLegend() {
  const el = $('legend');
  el.innerHTML = neurons.meta.classes
    .filter((c) => c.count > 0)
    .map((c) => `<div><i style="background:${c.color}"></i>${c.name} <span style="opacity:.6">${c.count.toLocaleString()}</span></div>`)
    .join('');
}

function wireControls() {
  $('btn-run').onclick = () => send({ cmd: 'run', on: !running });
  $('btn-reset').onclick = () => { lastEvents = null; send({ cmd: 'reset' }); };
  $('sel-mode').onchange = (e) => { lastEvents = null; send({ cmd: 'mode', mode: e.target.value }); };
  $('rng-drive').oninput = (e) => {
    $('out-drive').textContent = e.target.value;
    send({ cmd: 'set', key: 'drive', value: e.target.value / 100 });
  };
  $('rng-assist').oninput = (e) => {
    $('out-assist').textContent = `${e.target.value}%`;
    send({ cmd: 'set', key: 'assist', value: e.target.value / 100 });
  };
  $('sel-speed').onchange = (e) => send({ cmd: 'set', key: 'speed', value: parseFloat(e.target.value) });
  $('chk-rotate').onchange = (e) => brain.setAutoRotate(e.target.checked);
  $('chk-raw-spikes').onchange = (e) => brain.setRawSpikes(e.target.checked);
  $('rng-bloom').oninput = (e) => brain.setBloom(e.target.value / 100);
  $('rng-size').oninput = (e) => brain.setPointSize(e.target.value / 100);
  $('sel-clones').onchange = (e) => brain.setClones(parseInt(e.target.value, 10));
  document.querySelectorAll('.cam-buttons button').forEach((b) => {
    b.onclick = () => {
      document.querySelectorAll('.cam-buttons button').forEach((x) => x.classList.toggle('on', x === b));
      kart.setCam(b.dataset.cam);
    };
  });

  // Hover tooltip + click-to-poke on the brain.
  const canvas = $('brain-canvas');
  const tip = $('tooltip');
  let lastPick = 0;
  let downAt = null;
  canvas.addEventListener('pointermove', (ev) => {
    const t = performance.now();
    if (t - lastPick < 80) return;
    lastPick = t;
    const i = brain.pick(ev.clientX, ev.clientY);
    if (i < 0) { tip.classList.add('hidden'); return; }
    const r = canvas.getBoundingClientRect();
    const typeName = neurons.meta.types[neurons.type[i]] || '(untyped)';
    const cls = neurons.meta.classes[neurons.cls[i]].name;
    const side = ['', 'left', 'right', 'center'][neurons.side[i]] || '';
    tip.innerHTML = `<b>${typeName}</b> <span style="opacity:.7">${side}</span><br>${cls} · #${i}<br><span style="opacity:.6">click to poke all ${typeName}</span>`;
    tip.style.left = `${Math.min(r.width - 200, ev.clientX - r.left + 14)}px`;
    tip.style.top = `${ev.clientY - r.top + 14}px`;
    tip.classList.remove('hidden');
  });
  canvas.addEventListener('pointerleave', () => tip.classList.add('hidden'));
  canvas.addEventListener('pointerdown', (ev) => { downAt = [ev.clientX, ev.clientY]; });
  canvas.addEventListener('pointerup', (ev) => {
    if (!downAt || Math.hypot(ev.clientX - downAt[0], ev.clientY - downAt[1]) > 4) return;
    const i = brain.pick(ev.clientX, ev.clientY);
    if (i >= 0) send({ cmd: 'pokeNeuron', index: i });
  });
}

let lastT = performance.now() / 1000;
function loop() {
  const t0 = performance.now();
  const now = t0 / 1000;
  const dt = Math.min(0.1, now - lastT);
  lastT = now;

  const ki = kart.render(dt, now);
  const kartCalls = ki.calls;
  const bi = brain.render(now);
  raster.draw();
  eyePanel.draw(lastEye);

  frameStats.n++;
  frameStats.cpu += performance.now() - t0;
  frameStats.upload += brain.uploadBytes;
  const el = t0 - frameStats.t0;
  if (el > 500) {
    const fps = (frameStats.n * 1000) / el;
    perf.set('fps', fps.toFixed(0));
    perf.set('frame', (frameStats.cpu / frameStats.n).toFixed(2));
    perf.set('points', bi.points.toLocaleString());
    perf.set('calls', `${bi.calls + kartCalls}`);
    perf.set('upload', `${((frameStats.upload * 1000) / el / 1e6).toFixed(1)} MB`);
    perf.set('ws', `${((net.bytes * 1000) / el / 1e3).toFixed(0)} KB/s · ${Math.round((net.msgs * 1000) / el)}/s`);
    perf.pushFps(fps);
    frameStats.n = 0; frameStats.cpu = 0; frameStats.upload = 0; frameStats.t0 = t0;
    net.bytes = 0; net.msgs = 0;
  }
  requestAnimationFrame(loop);
}

async function main() {
  perf = new PerfPanel($('perf-grid'), $('perf-canvas'));
  try {
    neurons = await loadNeurons();
  } catch (err) {
    $('subtitle').textContent = 'No neuron data yet: run `uv run flykart prepare` first.';
    throw err;
  }
  brain = new BrainView($('brain-canvas'), $('brain-canvas').parentElement, neurons);
  setConn(false);
  buildLegend();
  wireControls();
  connect();
}

main();
