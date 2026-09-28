// Optional artist models. List .glb files in web/assets/models/models.json and
// they replace the built-in primitive models (kart, truck, obstacle, sugar) or
// get placed as scenery. Anything missing falls back to the primitives, so the
// game always runs. See web/assets/models/README.md for the format.
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { clone as cloneSkinned } from 'three/addons/utils/SkeletonUtils.js';

const BASE = '/assets/models/';
const DEG = Math.PI / 180;

export async function loadModels() {
  let manifest;
  try {
    const r = await fetch(`${BASE}models.json`, { cache: 'no-cache' });
    if (!r.ok) return [];
    manifest = await r.json();
  } catch {
    return [];
  }
  const loader = new GLTFLoader();
  const loaded = [];
  for (const entry of manifest.models || []) {
    try {
      const gltf = await loader.loadAsync(BASE + entry.file);
      loaded.push({ ...entry, gltf });
    } catch (err) {
      console.warn(`FlyKart: could not load model ${entry.file}`, err);
    }
  }
  return loaded;
}

// A ready-to-place copy of a model: rotated as the manifest says, scaled so its
// widest horizontal side is `size` world units (the kart is ~2.6 long), centred
// on the origin and standing on the ground. Animated models get a mixer.
export function instance(entry) {
  const root = new THREE.Group();
  const obj = cloneSkinned(entry.gltf.scene);
  const [rx, ry, rz] = entry.rotate || [0, 0, 0];
  obj.rotation.set(rx * DEG, ry * DEG, rz * DEG);
  root.add(obj);
  obj.updateMatrixWorld(true);
  const box = new THREE.Box3().setFromObject(obj);
  const size = box.getSize(new THREE.Vector3());
  const s = entry.size ? entry.size / Math.max(size.x, size.z, 1e-6) : entry.scale ?? 1;
  obj.scale.multiplyScalar(s);
  obj.updateMatrixWorld(true);
  box.setFromObject(obj);
  const c = box.getCenter(new THREE.Vector3());
  obj.position.sub(new THREE.Vector3(c.x, box.min.y - (entry.lift || 0), c.z));
  obj.traverse((o) => {
    if (o.isMesh) {
      o.castShadow = true;
      o.receiveShadow = true;
    }
  });
  if (entry.gltf.animations?.length) {
    const mixer = new THREE.AnimationMixer(obj);
    for (const clip of entry.gltf.animations) mixer.clipAction(clip).play();
    root.userData.mixer = mixer;
  }
  return root;
}
