// The connectome as a point cloud: one point per neuron at its soma (or a
// representative point for neurons whose soma is outside the imaged volume).
// Activity updates on simulation packets; the shader smooths it between them.
// Raw batch flashes remain available as a separate display mode.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';
import { RecentActivity } from './presentation.js';

// Thousands of overlapping additive points can sum far above 1; squash that
// smoothly before bloom so a busy optic lobe glows instead of whiting out.
const CompressShader = {
  uniforms: { tDiffuse: { value: null }, uExposure: { value: 1.0 } },
  vertexShader: `varying vec2 vUv; void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
  fragmentShader: `uniform sampler2D tDiffuse; uniform float uExposure; varying vec2 vUv;
    void main() { vec4 c = texture2D(tDiffuse, vUv); gl_FragColor = vec4(1.0 - exp(-c.rgb * uExposure), 1.0); }`,
};

const VERT = /* glsl */ `
  attribute float aClass;
  attribute float aLast;
  attribute float aFrom;
  attribute float aActivity;
  attribute float aKey;
  uniform float uTime;
  uniform float uDecay;
  uniform float uStart;
  uniform float uDuration;
  uniform bool uRaw;
  uniform float uSize;
  uniform float uScale;
  uniform float uDim;
  uniform float uGain;
  uniform vec3 uColors[16];
  uniform vec3 uHot;
  varying vec3 vColor;
  varying float vAlpha;
  void main() {
    float p = clamp((uTime - uStart) / uDuration, 0.0, 1.0);
    float stale = exp(-max(uTime - uStart - uDuration, 0.0) / (2.0 * uDuration));
    float act = uRaw ? exp(-max(uTime - aLast, 0.0) / uDecay)
                    : mix(aFrom, aActivity, p) * stale;
    vec3 base = uColors[int(aClass + 0.5)];
    vec3 lit = mix(base, uHot, 0.35) * (uGain + 2.0 * aKey);
    vColor = mix(base * uDim * (1.0 + 4.0 * aKey), lit, act);
    vAlpha = mix(0.5, 0.8 + 0.2 * aKey, act);
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mv;
    float s = uSize * (1.0 + (0.8 + 1.5 * aKey) * act) * (1.0 + 1.2 * aKey);
    gl_PointSize = clamp(s * uScale / -mv.z, 1.0, 64.0);
  }
`;

const FRAG = /* glsl */ `
  varying vec3 vColor;
  varying float vAlpha;
  void main() {
    vec2 c = gl_PointCoord - 0.5;
    float r = length(c);
    if (r > 0.5) discard;
    float soft = smoothstep(0.5, 0.05, r);
    float core = smoothstep(0.2, 0.0, r);
    gl_FragColor = vec4(vColor * (soft + 0.8 * core), soft * vAlpha);
  }
`;

export class BrainView {
  constructor(canvas, container, neurons) {
    this.canvas = canvas;
    this.container = container;
    this.neurons = neurons;
    this.n = neurons.n;

    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: false, powerPreference: 'high-performance' });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.info.autoReset = false; // count every pass of the composer
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x05070d);
    this.camera = new THREE.PerspectiveCamera(38, 1, 0.01, 200);
    this.baseDistance = 3.1;
    // Three-quarter view from the front-left, so the nerve cord trailing behind
    // and below the brain is visible from the start.
    this.target = new THREE.Vector3(0, 0.3, 0.1);
    this.camera.position.copy(this.target).add(new THREE.Vector3(-1.5, 0.75, 2.6).setLength(this.baseDistance));
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.target.copy(this.target);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.autoRotate = true;
    this.controls.autoRotateSpeed = 0.5;

    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(neurons.pos, 3));
    g.setAttribute('aClass', new THREE.BufferAttribute(Float32Array.from(neurons.cls), 1));
    this.activity = new RecentActivity(this.n);
    this.last = this.activity.lastSpike;
    this.lastAttr = new THREE.BufferAttribute(this.last, 1);
    this.lastAttr.setUsage(THREE.DynamicDrawUsage);
    g.setAttribute('aLast', this.lastAttr);
    this.fromAttr = new THREE.BufferAttribute(this.activity.from, 1).setUsage(THREE.DynamicDrawUsage);
    this.activityAttr = new THREE.BufferAttribute(this.activity.target, 1).setUsage(THREE.DynamicDrawUsage);
    g.setAttribute('aFrom', this.fromAttr);
    g.setAttribute('aActivity', this.activityAttr);
    this.key = new Float32Array(this.n);
    this.keyAttr = new THREE.BufferAttribute(this.key, 1);
    g.setAttribute('aKey', this.keyAttr);
    g.computeBoundingSphere();
    this.geometry = g;

    const colors = new Array(16).fill(0).map(() => new THREE.Color(0x3a4a6a));
    neurons.meta.classes.forEach((c, i) => { if (i < 16) colors[i] = new THREE.Color(c.color); });
    this.material = new THREE.ShaderMaterial({
      uniforms: {
        uTime: { value: 0 },
        uDecay: { value: 0.22 },
        uStart: { value: 0 },
        uDuration: { value: 0.15 },
        uRaw: { value: false },
        uSize: { value: 0.012 },
        uScale: { value: 400 },
        uDim: { value: 0.05 },
        uGain: { value: 0.35 },
        uColors: { value: colors },
        uHot: { value: new THREE.Color(1.0, 0.95, 0.8) },
      },
      vertexShader: VERT,
      fragmentShader: FRAG,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    this.points = new THREE.Points(g, this.material);
    this.points.frustumCulled = false;
    this.cloneGroup = new THREE.Group();
    this.scene.add(this.points, this.cloneGroup);

    this.composer = new EffectComposer(this.renderer);
    this.composer.addPass(new RenderPass(this.scene, this.camera));
    this.composer.addPass(new ShaderPass(CompressShader));
    this.bloom = new UnrealBloomPass(new THREE.Vector2(256, 256), 0.8, 0.5, 0.45);
    this.composer.addPass(this.bloom);
    this.composer.addPass(new OutputPass());

    this.raycaster = new THREE.Raycaster();
    this.raycaster.params.Points.threshold = 0.012;
    this.pointer = new THREE.Vector2();
    this.labels = [];
    this.clones = 1;
    this.uploadBytes = 0;
    this._dirty = false;
    this._resize();
    new ResizeObserver(() => this._resize()).observe(container);
  }

  _resize() {
    const w = Math.max(1, this.container.clientWidth);
    const h = Math.max(1, this.container.clientHeight);
    this.renderer.setSize(w, h, false);
    this.composer.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.material.uniforms.uScale.value = h * this.renderer.getPixelRatio() * 0.9;
  }

  // Key neurons (the ones the kart reads or writes) render bigger and get labels.
  setGroups(groups, labelsEl) {
    this.key.fill(0);
    labelsEl.innerHTML = '';
    this.labels = [];
    for (const grp of groups) {
      if (!grp.indices.length) continue;
      for (const i of grp.indices) this.key[i] = 1;
      if (!grp.label) continue;
      const c = new THREE.Vector3();
      for (const i of grp.indices) c.add(new THREE.Vector3(this.neurons.pos[3 * i], this.neurons.pos[3 * i + 1], this.neurons.pos[3 * i + 2]));
      c.divideScalar(grp.indices.length);
      const el = document.createElement('div');
      el.className = 'label';
      el.textContent = grp.label;
      el.style.color = grp.color;
      labelsEl.appendChild(el);
      this.labels.push({ el, pos: c, indices: grp.indices, hotUntil: 0 });
    }
    this.keyAttr.needsUpdate = true;
  }

  // Called once per simulation frame with the neurons that spiked.
  addSpikes(idx, now, simSeconds, duration) {
    this.activity.update(idx, now, simSeconds, duration);
    this.material.uniforms.uStart.value = this.activity.start;
    this.material.uniforms.uDuration.value = this.activity.duration;
    this._dirty = true;
    const fired = new Set(idx);
    for (const lab of this.labels) {
      for (const i of lab.indices) if (fired.has(i)) { lab.hotUntil = now + duration; break; }
    }
  }

  resetActivity() {
    this.activity.reset();
    this._dirty = true;
    for (const lab of this.labels) lab.hotUntil = 0;
  }

  setRawSpikes(on) { this.material.uniforms.uRaw.value = on; }
  setBloom(v) { this.bloom.strength = v; }
  setPointSize(v) { this.material.uniforms.uSize.value = 0.012 * v; }
  setAutoRotate(on) { this.controls.autoRotate = on; }

  setClones(k) {
    this.clones = k;
    this.cloneGroup.clear();
    const side = Math.round(Math.sqrt(k));
    const spacing = 2.3;
    if (k > 1) {
      for (let a = 0; a < side; a++) {
        for (let b = 0; b < side; b++) {
          const x = (a - (side - 1) / 2) * spacing;
          const y = (b - (side - 1) / 2) * spacing;
          if (Math.abs(x) < 1e-6 && Math.abs(y) < 1e-6) continue;
          const p = new THREE.Points(this.geometry, this.material);
          p.position.set(x, y, 0);
          p.frustumCulled = false;
          this.cloneGroup.add(p);
        }
      }
    }
    // Even grids have no centre cell: shift the original too.
    const off = side % 2 === 0 ? spacing / 2 : 0;
    this.points.position.set(-off, -off, 0);
    const d = this.baseDistance * Math.max(1, side * 1.25);
    const dir = this.camera.position.clone().sub(this.controls.target).setLength(d);
    this.camera.position.copy(this.controls.target).add(dir);
    this.controls.update();
  }

  pick(clientX, clientY) {
    const r = this.canvas.getBoundingClientRect();
    this.pointer.set(((clientX - r.left) / r.width) * 2 - 1, -((clientY - r.top) / r.height) * 2 + 1);
    this.raycaster.setFromCamera(this.pointer, this.camera);
    this.raycaster.params.Points.threshold = 0.012 * (this.camera.position.distanceTo(this.controls.target) / this.baseDistance);
    const hits = this.raycaster.intersectObject(this.points, false);
    return hits.length ? hits[0].index : -1;
  }

  render(now) {
    this.material.uniforms.uTime.value = now;
    if (this._dirty) {
      this.lastAttr.needsUpdate = true;
      this.fromAttr.needsUpdate = true;
      this.activityAttr.needsUpdate = true;
      this.uploadBytes = this.last.byteLength + this.activity.from.byteLength + this.activity.target.byteLength;
      this._dirty = false;
    } else {
      this.uploadBytes = 0;
    }
    this.controls.update();
    this.renderer.info.reset();
    this.composer.render();

    // Project labels, then nudge overlapping ones apart vertically.
    const w = this.container.clientWidth, h = this.container.clientHeight;
    const v = new THREE.Vector3();
    const placed = [];
    for (const lab of this.labels) {
      v.copy(lab.pos).add(this.points.position).project(this.camera);
      lab.visible = v.z < 1 && Math.abs(v.x) < 1.05 && Math.abs(v.y) < 1.05;
      lab.sx = ((v.x + 1) / 2) * w;
      lab.sy = ((1 - v.y) / 2) * h;
    }
    const order = this.labels.filter((l) => l.visible).sort((a, b) => a.sy - b.sy);
    for (const lab of order) {
      const halfW = lab.el.offsetWidth / 2 + 3;
      let y = lab.sy;
      for (const p of placed) {
        if (Math.abs(p.x - lab.sx) < p.halfW + halfW && Math.abs(p.y - y) < 17) y = p.y + 17;
      }
      placed.push({ x: lab.sx, y, halfW });
      lab.el.style.left = `${lab.sx}px`;
      lab.el.style.top = `${y}px`;
    }
    for (const lab of this.labels) {
      lab.el.style.display = lab.visible ? '' : 'none';
      lab.el.classList.toggle('hot', now < lab.hotUntil);
    }
    return this.renderer.info.render;
  }
}
