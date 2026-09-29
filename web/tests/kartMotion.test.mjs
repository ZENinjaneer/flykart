import test from 'node:test';
import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';

// Use the browser's existing import map for Three.js, without a renderer.
registerHooks({
  resolve(specifier, context, nextResolve) {
    if (specifier === 'three') return { url: new URL('../vendor/three/build/three.module.js', import.meta.url).href, shortCircuit: true };
    if (specifier.startsWith('three/addons/')) {
      return { url: new URL(`../vendor/three/examples/jsm/${specifier.slice(13)}`, import.meta.url).href, shortCircuit: true };
    }
    return nextResolve(specifier, context);
  },
});
const THREE = await import('three');
const { KartView } = await import('../js/kartView.js');

function fixture() {
  const view = Object.create(KartView.prototype);
  Object.assign(view, {
    state: null, disp: { x: 0, y: 0, h: 0, z: 0, t: 0 }, lead: { x: 0, y: 0, h: 0 },
    transition: null, lastRendered: null, running: true, wheelSpin: 0,
    obstacles: new Map(), sugars: new Map(), truck: new THREE.Object3D(),
    kart: new THREE.Object3D(), wings: [new THREE.Object3D(), new THREE.Object3D()],
    wheels: [new THREE.Object3D()], proboscis: new THREE.Object3D(), mixers: [],
    sun: new THREE.Object3D(), camera: new THREE.PerspectiveCamera(), camMode: 'top',
    renderer: { render() {}, info: { render: {} } }, scene: new THREE.Scene(),
  });
  view.sun.target = new THREE.Object3D();
  return view;
}
const state = (t, x) => ({ t, kart: { x, y: 0, z: 0, h: 0, v: 6, steer: 0, feed: 0 }, lead: null, obstacles: [], sugars: [] });

test('pause reaches latest known state and snapshot cannot restart interpolation', () => {
  const v = fixture();
  v.update(state(0, 0), 0, 0.6);
  v.update(state(0.0144, 6), 1, 0.6);
  v._sample(1.3);
  assert.ok(Math.abs(v.disp.x - 3) < 1e-10);
  v.setRunning(false);
  v.update(state(0.0144, 6), 1.3, 0.6);
  assert.equal(v.disp.x, 6);
  v.render(0.1, 1.3);
  const wing = v.wings[0].rotation.z;
  const wheel = v.wheels[0].rotation.z;
  v.render(0.1, 99);
  assert.equal(v.disp.x, 6);
  assert.equal(v.wings[0].rotation.z, wing);
  assert.equal(v.wheels[0].rotation.z, wheel);
});

test('stalled running display stops at latest state and reset snaps to new origin', () => {
  const v = fixture();
  v.update(state(0, 0), 0, 0.6);
  v.render(0.1, 0);
  v.update(state(0.0144, 6), 1, 0.6);
  v.render(0.1, 2);
  const wheel = v.wheels[0].rotation.z;
  const wing = v.wings[0].rotation.z;
  v.render(0.1, 99);
  assert.equal(v.disp.x, 6);
  assert.equal(v.wheels[0].rotation.z, wheel);
  assert.equal(v.wings[0].rotation.z, wing);
  v.reset();
  v.update(state(0, 100), 100, 0.6);
  assert.equal(v.disp.x, 100);
  assert.equal(v.disp.t, 0);
});
