// The kart world in Three.js. The server owns the physics; this file only
// interpolates between received states, stopping when the latest is reached.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { instance } from './models.js';
import { interpolatePose, progress } from './presentation.js';

// Server coordinates are (x, y) on the ground with heading counter-clockwise
// from +x. In Three.js y is up, so world (x, y) -> (x, 0, -y) and a model
// built facing +x just needs rotation.y = heading.
const toVec = (x, y, z = 0) => new THREE.Vector3(x, z, -y);

function asphaltTexture() {
  const c = document.createElement('canvas');
  c.width = 128; c.height = 256;
  const g = c.getContext('2d');
  g.fillStyle = '#2b2f36'; g.fillRect(0, 0, 128, 256);
  for (let i = 0; i < 900; i++) {
    const v = 36 + Math.random() * 30;
    g.fillStyle = `rgb(${v},${v + 2},${v + 6})`;
    g.fillRect(Math.random() * 128, Math.random() * 256, 1.5, 1.5);
  }
  g.fillStyle = '#e8e8e8';
  g.fillRect(4, 0, 4, 256); g.fillRect(120, 0, 4, 256);
  g.fillStyle = '#f5d142';
  g.fillRect(62, 0, 4, 120);
  const t = new THREE.CanvasTexture(c);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.anisotropy = 8;
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

function stripeTexture() {
  const c = document.createElement('canvas');
  c.width = 64; c.height = 64;
  const g = c.getContext('2d');
  for (let i = 0; i < 4; i++) {
    g.fillStyle = i % 2 ? '#1b120c' : '#8a5a2b';
    g.fillRect(i * 16, 0, 16, 64);
  }
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

function makeFlyKart() {
  const kart = new THREE.Group();
  const metal = new THREE.MeshStandardMaterial({ color: 0x20242c, metalness: 0.6, roughness: 0.35 });
  const chassis = new THREE.Mesh(new THREE.BoxGeometry(2.6, 0.35, 1.5), metal);
  chassis.position.y = 0.45;
  kart.add(chassis);

  const thoraxMat = new THREE.MeshStandardMaterial({ color: 0x6b4a2a, roughness: 0.6 });
  const thorax = new THREE.Mesh(new THREE.SphereGeometry(0.55, 24, 16), thoraxMat);
  thorax.scale.set(1.1, 0.85, 0.95);
  thorax.position.set(0.35, 0.95, 0);
  const abdomen = new THREE.Mesh(
    new THREE.SphereGeometry(0.6, 24, 16),
    new THREE.MeshStandardMaterial({ map: stripeTexture(), roughness: 0.55 }),
  );
  abdomen.scale.set(1.5, 0.75, 0.85);
  abdomen.position.set(-0.75, 0.9, 0);
  const head = new THREE.Mesh(new THREE.SphereGeometry(0.38, 20, 14), thoraxMat);
  head.position.set(1.05, 1.02, 0);
  kart.add(thorax, abdomen, head);

  const eyeMat = new THREE.MeshStandardMaterial({ color: 0xc0121e, roughness: 0.25, emissive: 0x3a0005 });
  for (const s of [-1, 1]) {
    const eye = new THREE.Mesh(new THREE.SphereGeometry(0.3, 24, 16), eyeMat);
    eye.position.set(1.12, 1.1, 0.3 * s);
    eye.scale.set(0.9, 1.05, 0.8);
    kart.add(eye);
    const ant = new THREE.Mesh(new THREE.CylinderGeometry(0.025, 0.02, 0.4), thoraxMat);
    ant.position.set(1.35, 1.3, 0.1 * s);
    ant.rotation.z = -0.9;
    kart.add(ant);
  }
  const proboscis = new THREE.Mesh(
    new THREE.CylinderGeometry(0.06, 0.1, 1, 10),
    new THREE.MeshStandardMaterial({ color: 0x9a6a3a, roughness: 0.5 }),
  );
  proboscis.geometry.translate(0, -0.5, 0);
  proboscis.position.set(1.25, 0.9, 0);
  proboscis.rotation.z = 0.5;
  kart.add(proboscis);

  const wingMat = new THREE.MeshStandardMaterial({
    color: 0xcfe8ff, transparent: true, opacity: 0.35, side: THREE.DoubleSide, metalness: 0.2, roughness: 0.1,
    emissive: 0x223344,
  });
  const wings = [];
  for (const s of [-1, 1]) {
    const w = new THREE.Mesh(new THREE.CircleGeometry(0.9, 24), wingMat);
    w.scale.set(1.3, 0.42, 1);
    w.position.set(-0.6, 1.25, 0.5 * s);
    w.rotation.set(-Math.PI / 2 + 0.25 * s, 0, -0.35 * s);
    kart.add(w);
    wings.push(w);
  }

  const wheelMat = new THREE.MeshStandardMaterial({ color: 0x111111, roughness: 0.8 });
  const hubMat = new THREE.MeshStandardMaterial({ color: 0xd0d6e0, metalness: 0.8, roughness: 0.3 });
  const wheels = [];
  for (const [x, z] of [[0.9, 0.85], [0.9, -0.85], [-0.9, 0.9], [-0.9, -0.9]]) {
    const wg = new THREE.Group();
    const tire = new THREE.Mesh(new THREE.CylinderGeometry(0.38, 0.38, 0.32, 20), wheelMat);
    tire.rotation.x = Math.PI / 2;
    const hub = new THREE.Mesh(new THREE.CylinderGeometry(0.16, 0.16, 0.34, 12), hubMat);
    hub.rotation.x = Math.PI / 2;
    wg.add(tire, hub);
    wg.position.set(x, 0.38, z);
    kart.add(wg);
    wheels.push(wg);
  }
  kart.traverse((o) => { if (o.isMesh) { o.castShadow = true; } });
  return { kart, wheels, wings, proboscis };
}

function makeSugarTruck() {
  const g = new THREE.Group();
  const body = new THREE.Mesh(new THREE.BoxGeometry(2.4, 1.1, 1.6), new THREE.MeshStandardMaterial({ color: 0x3d7cff, roughness: 0.4 }));
  body.position.set(-0.3, 0.9, 0);
  const cab = new THREE.Mesh(new THREE.BoxGeometry(1.0, 1.3, 1.6), new THREE.MeshStandardMaterial({ color: 0xf2f4f8, roughness: 0.4 }));
  cab.position.set(1.3, 1.0, 0);
  const cube = new THREE.Mesh(new THREE.BoxGeometry(1.2, 1.2, 1.2), new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: 0x666666, roughness: 0.9 }));
  cube.position.set(-0.4, 2.05, 0);
  cube.rotation.y = 0.4;
  g.add(body, cab, cube);
  for (const [x, z] of [[1.1, 0.8], [1.1, -0.8], [-1.0, 0.8], [-1.0, -0.8]]) {
    const w = new THREE.Mesh(new THREE.CylinderGeometry(0.4, 0.4, 0.3, 16), new THREE.MeshStandardMaterial({ color: 0x111111 }));
    w.rotation.x = Math.PI / 2;
    w.position.set(x, 0.4, z);
    g.add(w);
  }
  g.traverse((o) => { if (o.isMesh) o.castShadow = true; });
  return g;
}

function makeSwatter() {
  const g = new THREE.Group();
  const stick = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.08, 2.2), new THREE.MeshStandardMaterial({ color: 0xf7c948 }));
  stick.position.y = 1.1;
  const headMat = new THREE.MeshStandardMaterial({ color: 0xff3355, transparent: true, opacity: 0.9, roughness: 0.4, side: THREE.DoubleSide });
  const head = new THREE.Mesh(new THREE.BoxGeometry(0.12, 1.5, 1.8), headMat);
  head.position.y = 2.4;
  const grid = new THREE.LineSegments(
    new THREE.EdgesGeometry(new THREE.BoxGeometry(0.14, 1.5, 1.8, 1, 5, 6)),
    new THREE.LineBasicMaterial({ color: 0x550010 }),
  );
  grid.position.copy(head.position);
  g.add(stick, head, grid);
  g.traverse((o) => { if (o.isMesh) o.castShadow = true; });
  return { group: g, headMat };
}

export class KartView {
  constructor(canvas, container, staticState) {
    this.canvas = canvas;
    this.container = container;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x9cc7ee);
    this.scene.fog = new THREE.Fog(0x9cc7ee, 60, 230);
    this.camera = new THREE.PerspectiveCamera(60, 1, 0.1, 600);
    this.camera.position.set(0, 12, 20);
    this.orbit = new OrbitControls(this.camera, canvas);
    this.orbit.enabled = false;
    this.camMode = 'chase';

    this.scene.add(new THREE.HemisphereLight(0xdfefff, 0x4a6b3a, 1.1));
    const sun = new THREE.DirectionalLight(0xfff1dc, 2.2);
    sun.position.set(40, 70, 25);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    const sc = sun.shadow.camera;
    sc.left = -30; sc.right = 30; sc.top = 30; sc.bottom = -30; sc.near = 1; sc.far = 200;
    this.sun = sun;
    this.scene.add(sun, sun.target);

    const ground = new THREE.Mesh(
      new THREE.CircleGeometry(400, 64),
      new THREE.MeshStandardMaterial({ color: 0x5f8f4a, roughness: 1 }),
    );
    ground.rotation.x = -Math.PI / 2;
    ground.receiveShadow = true;
    this.scene.add(ground);

    this._buildTrack(staticState.track, staticState.halfWidth);

    const fk = makeFlyKart();
    this.kart = fk.kart; this.wheels = fk.wheels; this.wings = fk.wings; this.proboscis = fk.proboscis;
    this.scene.add(this.kart);
    this.truck = makeSugarTruck();
    this.scene.add(this.truck);

    this.obstacles = new Map();
    this.sugars = new Map();
    this.sugarGeo = new THREE.BoxGeometry(0.9, 0.9, 0.9);
    this.sugarMat = new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: 0xbfd9ff, emissiveIntensity: 0.6, roughness: 0.7 });

    this.obstacleModel = null;
    this.sugarModel = null;
    this.mixers = [];
    this.state = null;
    this.disp = { x: 0, y: 0, h: 0, z: 0, t: 0 };
    this.lead = { x: 0, y: 0, h: 0 };
    this.running = false;
    this.transition = null;
    this.lastRendered = null;
    this.wheelSpin = 0;
    this._resize();
    new ResizeObserver(() => this._resize()).observe(container);
  }

  _buildTrack(pts, hw) {
    const n = pts.length;
    const pos = [], uv = [], idx = [];
    let along = 0;
    for (let i = 0; i <= n; i++) {
      const p = pts[i % n], q = pts[(i + 1) % n], o = pts[(i - 1 + n) % n];
      const tx = q[0] - o[0], ty = q[1] - o[1];
      const l = Math.hypot(tx, ty) || 1;
      const nx = -ty / l, ny = tx / l;
      if (i > 0) along += Math.hypot(p[0] - pts[i - 1][0], p[1] - pts[i - 1][1]);
      const L = toVec(p[0] + nx * hw, p[1] + ny * hw, 0.02);
      const R = toVec(p[0] - nx * hw, p[1] - ny * hw, 0.02);
      pos.push(L.x, L.y, L.z, R.x, R.y, R.z);
      uv.push(0, along / 10, 1, along / 10);
      if (i < n) {
        const a = 2 * i;
        idx.push(a, a + 1, a + 2, a + 1, a + 3, a + 2);
      }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
    g.setIndex(idx);
    g.computeVertexNormals();
    const road = new THREE.Mesh(g, new THREE.MeshStandardMaterial({ map: asphaltTexture(), roughness: 0.9, side: THREE.DoubleSide }));
    road.receiveShadow = true;
    this.scene.add(road);

    // Curbs: one InstancedMesh per colour, alternating along both edges.
    const curbGeo = new THREE.BoxGeometry(1.6, 0.25, 0.7);
    const red = new THREE.InstancedMesh(curbGeo, new THREE.MeshStandardMaterial({ color: 0xe23b3b }), n);
    const white = new THREE.InstancedMesh(curbGeo, new THREE.MeshStandardMaterial({ color: 0xf2f2f2 }), n);
    let ri = 0, wi = 0;
    const m = new THREE.Matrix4(), qn = new THREE.Quaternion(), sc = new THREE.Vector3(1, 1, 1);
    for (let i = 0; i < n; i += 1) {
      const p = pts[i], q = pts[(i + 1) % n];
      const h = Math.atan2(q[1] - p[1], q[0] - p[0]);
      const nx = -Math.sin(h), ny = Math.cos(h);
      const side = i % 2 === 0 ? 1 : -1;
      const c = toVec(p[0] + nx * (hw + 0.35) * side, p[1] + ny * (hw + 0.35) * side, 0.12);
      qn.setFromAxisAngle(new THREE.Vector3(0, 1, 0), h);
      m.compose(c, qn, sc);
      if (Math.floor(i / 2) % 2 === 0) red.setMatrixAt(ri++, m); else white.setMatrixAt(wi++, m);
    }
    red.count = ri; white.count = wi;
    this.scene.add(red, white);

    // A few trees for depth cues.
    const trunkGeo = new THREE.CylinderGeometry(0.3, 0.4, 2.5);
    const leafGeo = new THREE.ConeGeometry(2.0, 5, 8);
    const trunks = new THREE.InstancedMesh(trunkGeo, new THREE.MeshStandardMaterial({ color: 0x6b4a2a }), 120);
    const leaves = new THREE.InstancedMesh(leafGeo, new THREE.MeshStandardMaterial({ color: 0x2f6b3a, roughness: 0.9 }), 120);
    let k = 0;
    let seed = 7;
    const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
    while (k < 120) {
      const x = (rnd() - 0.5) * 260, y = (rnd() - 0.5) * 260;
      let near = false;
      for (let i = 0; i < n; i += 4) if (Math.hypot(pts[i][0] - x, pts[i][1] - y) < hw + 5) { near = true; break; }
      if (near) continue;
      const s = 0.7 + rnd() * 0.9;
      m.compose(toVec(x, y, 1.25 * s), new THREE.Quaternion(), new THREE.Vector3(s, s, s));
      trunks.setMatrixAt(k, m);
      m.compose(toVec(x, y, (2.5 + 2.5) * s), new THREE.Quaternion(), new THREE.Vector3(s, s, s));
      leaves.setMatrixAt(k, m);
      k++;
    }
    trunks.castShadow = leaves.castShadow = true;
    this.scene.add(trunks, leaves);
  }

  _resize() {
    const w = Math.max(1, this.container.clientWidth);
    const h = Math.max(1, this.container.clientHeight);
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  setCam(mode) {
    this.camMode = mode;
    this.orbit.enabled = mode === 'orbit';
    if (mode === 'orbit') {
      this.orbit.target.copy(this.kart.position);
      this.camera.position.set(this.kart.position.x + 25, 30, this.kart.position.z + 25);
    }
  }

  // Swap in artist models from web/assets/models/models.json (see models.js).
  // Returns the credit lines to show.
  applyModels(models) {
    const credits = [];
    const adopt = (group, entry) => {
      for (const c of group.children) c.visible = false;
      const m = instance(entry);
      group.add(m);
      if (m.userData.mixer) this.mixers.push(m.userData.mixer);
    };
    for (const entry of models) {
      if (entry.role === 'kart') adopt(this.kart, entry);
      else if (entry.role === 'truck') adopt(this.truck, entry);
      else if (entry.role === 'obstacle') this.obstacleModel = entry;
      else if (entry.role === 'sugar') this.sugarModel = entry;
      else if (entry.role === 'scenery') {
        for (const [x, y, heading = 0] of entry.place || [[0, 0, 0]]) {
          const m = instance(entry);
          m.position.copy(toVec(x, y));
          m.rotation.y = heading * (Math.PI / 180);
          this.scene.add(m);
          if (m.userData.mixer) this.mixers.push(m.userData.mixer);
        }
      }
      if (entry.credit && !credits.includes(entry.credit)) credits.push(entry.credit);
    }
    // Rebuild obstacles / sugar on the next update so they use the new models.
    for (const e of this.obstacles.values()) this.scene.remove(e.group);
    for (const e of this.sugars.values()) this.scene.remove(e);
    this.obstacles.clear();
    this.sugars.clear();
    return credits;
  }

  setRunning(on) {
    this.running = on;
    if (!on && this.transition) {
      this._sample(this.transition.start + this.transition.duration);
      this.transition.duration = 0;
    }
  }

  reset() {
    this.state = null;
    this.transition = null;
    this.wheelSpin = 0;
    this.lastRendered = null;
  }

  _sample(now) {
    if (!this.transition) return;
    const tr = this.transition;
    const a = progress(now, tr.start, tr.duration);
    this.disp = interpolatePose(tr.from, tr.to, a);
    if (tr.leadTo) this.lead = interpolatePose(tr.leadFrom, tr.leadTo, a);
  }

  update(state, now, duration) {
    this._sample(now);
    if (!this.state || state.t !== this.state.t) {
      const snap = !this.state || !this.running || state.t < this.state.t ||
        Math.hypot(state.kart.x - this.disp.x, state.kart.y - this.disp.y) > 15;
      const to = { ...state.kart, t: state.t };
      this.transition = {
        from: snap ? to : { ...this.disp }, to,
        leadFrom: snap || !this.state?.lead ? state.lead : { ...this.lead },
        leadTo: state.lead,
        start: now, duration: snap ? 0 : duration,
      };
      this._sample(now);
    }
    this.state = state;
    // Obstacles
    for (const o of state.obstacles) {
      let e = this.obstacles.get(o.id);
      if (!e) {
        e = this.obstacleModel ? { group: instance(this.obstacleModel), headMat: null } : makeSwatter();
        if (e.group.userData.mixer) this.mixers.push(e.group.userData.mixer);
        this.scene.add(e.group);
        this.obstacles.set(o.id, e);
      }
      e.group.position.copy(toVec(o.x, o.y));
      e.group.visible = !o.ok;
      e.group.rotation.z = o.hit ? -1.3 : 0;
      if (e.headMat) e.headMat.color.set(o.hit ? 0x772233 : 0xff3355);
    }
    for (const s of state.sugars) {
      let e = this.sugars.get(s.id);
      if (!e) {
        if (this.sugarModel) {
          e = instance(this.sugarModel);
        } else {
          e = new THREE.Mesh(this.sugarGeo, this.sugarMat);
          e.castShadow = true;
        }
        this.scene.add(e);
        this.sugars.set(s.id, e);
      }
      e.position.copy(toVec(s.x, s.y, this.sugarModel ? 0 : 0.9));
      e.visible = s.on;
    }
    this.truck.visible = !!state.lead;
  }

  render(dt, now) {
    const before = this.lastRendered || { ...this.disp };
    if (this.running) this._sample(now);
    const simDt = this.running ? Math.max(0, this.disp.t - before.t) : 0;
    for (const m of this.mixers) m.update(simDt);
    const s = this.state;
    if (s) {
      const k = s.kart;
      this.kart.position.copy(toVec(this.disp.x, this.disp.y, this.disp.z));
      this.kart.rotation.set(0, this.disp.h, this.disp.z > 0.05 ? 0.15 : 0);
      if (simDt > 0) {
        const distance = Math.hypot(this.disp.x - before.x, this.disp.y - before.y);
        this.wheelSpin -= Math.sign(k.v) * distance / 0.38;
      }
      this.wheels.forEach((w, i) => {
        w.rotation.z = this.wheelSpin;
        w.rotation.y = i < 2 ? -k.steer * 0.5 : 0;
      });
      const flap = Math.min(1, Math.abs(k.v) / 10) * Math.sin(this.disp.t * 40) * 0.25;
      this.wings.forEach((w, i) => { w.rotation.z = (i ? 0.35 : -0.35) + flap * (i ? 1 : -1); });
      this.proboscis.scale.y = 0.25 + 0.9 * k.feed;
      if (s.lead) {
        this.truck.position.copy(toVec(this.lead.x, this.lead.y));
        this.truck.rotation.y = this.lead.h;
      }
      for (const e of this.sugars.values()) {
        if (this.sugarModel) e.rotation.y = this.disp.t * 1.1;
        else e.rotation.set(this.disp.t * 0.7, this.disp.t * 1.1, 0);
      }

      // Keep the shadow camera on the kart.
      this.sun.position.set(this.kart.position.x + 40, 70, this.kart.position.z + 25);
      this.sun.target.position.copy(this.kart.position);

      const fwd = new THREE.Vector3(Math.cos(this.disp.h), 0, -Math.sin(this.disp.h));
      if (this.camMode === 'chase') {
        const want = this.kart.position.clone().addScaledVector(fwd, -9).add(new THREE.Vector3(0, 4.2, 0));
        this.camera.position.lerp(want, 1 - Math.exp(-dt / 0.18));
        this.camera.lookAt(this.kart.position.clone().addScaledVector(fwd, 5).add(new THREE.Vector3(0, 1.2, 0)));
      } else if (this.camMode === 'top') {
        const want = this.kart.position.clone().add(new THREE.Vector3(0, 70, 0.01));
        this.camera.position.lerp(want, 1 - Math.exp(-dt / 0.25));
        this.camera.lookAt(this.kart.position);
      } else {
        this.orbit.target.lerp(this.kart.position, 0.1);
        this.orbit.update();
      }
    }
    this.lastRendered = { ...this.disp };
    this.renderer.render(this.scene, this.camera);
    return this.renderer.info.render;
  }
}
