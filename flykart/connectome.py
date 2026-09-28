"""Download and preprocess the MaleCNS v1.0 connectome (Janelia / Google, 2026).

`prepare()` turns the official feather tables into
  data/processed/connectome.npz   CSR weights (rows = postsynaptic) + per-neuron sign
  data/processed/neurons.parquet  one row per neuron: ids, type, class, side, visual field
  web/data/neurons.{bin,json}     positions / classes / types for the Three.js view

Only neurons with status "Traced" are kept: 165,122 neurons and 25,563,197
neuron-to-neuron connections (124M synapses).
"""

from __future__ import annotations

import json
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
WEB_DATA = ROOT / "web" / "data"

BASE_URL = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"
FILES = {
    "annotations": "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "neurotransmitters": "body-neurotransmitters-male-cns-v1.0.feather",
    "weights": "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
}

# Display classes for the brain view: (name, colour, superclasses).
CLASSES = [
    ("other", "#46536f", []),
    ("sensory", "#ffcf5a", ["cb_sensory", "vnc_sensory", "sensory_ascending", "sensory_descending",
                            "cb_sensory_tbc", "vnc_sensory_tbc", "sensory_ascending_tbc"]),
    ("photoreceptor", "#fff2a8", ["ol_sensory"]),
    ("optic lobe", "#3f8cff", ["ol_intrinsic"]),
    ("visual projection", "#5ef1ff", ["visual_projection", "visual_projection_tbc"]),
    ("visual centrifugal", "#7d7dff", ["visual_centrifugal"]),
    ("central brain", "#b48cff", ["cb_intrinsic"]),
    ("descending", "#ff4fd8", ["descending_neuron", "descending_neuron_tbc", "efferent_descending"]),
    ("ascending", "#ff9a4f", ["ascending_neuron", "efferent_ascending"]),
    ("nerve cord", "#4fffa8", ["vnc_intrinsic", "vnc_tbc"]),
    ("motor / efferent", "#ff5468", ["vnc_motor", "cb_motor", "vnc_efferent", "cb_efferent"]),
    ("endocrine", "#c8ffef", ["cb_endocrine", "vnc_endocrine", "ENS"]),
]
INHIBITORY_NT = {"gaba", "glutamate"}  # Shiu et al. 2024 sign rule


def is_prepared() -> bool:
    return all(f.exists() for f in (PROCESSED / "connectome.npz", PROCESSED / "neurons.parquet",
                                    WEB_DATA / "neurons.bin", WEB_DATA / "neurons.json"))


def download(force: bool = False) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for name in FILES.values():
        dest = RAW / name
        if dest.exists() and not force:
            continue
        url = f"{BASE_URL}/{name}"
        tmp = dest.with_suffix(".part")
        print(f"downloading {name}")
        with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
            total = int(r.headers.get("Content-Length", 0))
            done = 0
            while chunk := r.read(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    sys.stdout.write(f"\r  {done / 1e6:8.1f} / {total / 1e6:.1f} MB")
                    sys.stdout.flush()
        print()
        tmp.rename(dest)


def _weighted_partner_mean(values: np.ndarray, src: np.ndarray, dst: np.ndarray, w: np.ndarray, n: int) -> np.ndarray:
    """For every neuron d: synapse-weighted mean of values[s] over edges s -> d (NaN-aware)."""
    v = values[src]
    ok = ~np.isnan(v).any(axis=1) if v.ndim == 2 else ~np.isnan(v)
    ww = w[ok].astype(np.float64)
    denom = np.bincount(dst[ok], weights=ww, minlength=n)
    if values.ndim == 1:
        num = np.bincount(dst[ok], weights=ww * v[ok], minlength=n)
        out = num / np.maximum(denom, 1e-9)
        out[denom == 0] = np.nan
        return out
    out = np.full((n, values.shape[1]), np.nan)
    for c in range(values.shape[1]):
        num = np.bincount(dst[ok], weights=ww * v[ok, c], minlength=n)
        out[:, c] = num / np.maximum(denom, 1e-9)
    out[denom == 0] = np.nan
    return out


def prepare() -> None:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.feather as pf

    download()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    WEB_DATA.mkdir(parents=True, exist_ok=True)

    print("reading annotations")
    ann = pd.read_feather(RAW / FILES["annotations"])
    ann = ann[ann.status == "Traced"].sort_values("bodyId").reset_index(drop=True)
    n = len(ann)
    body = ann.bodyId.to_numpy()

    print("reading neurotransmitters")
    nt = pd.read_feather(RAW / FILES["neurotransmitters"], columns=["body", "consensus_nt", "predicted_nt"])
    nt = nt.set_index("body").reindex(body)
    ntv = nt.consensus_nt.where(nt.consensus_nt.notna() & (nt.consensus_nt != "unclear"), nt.predicted_nt)
    ntv = ntv.fillna("unclear").to_numpy()
    sign = np.where(np.isin(ntv, list(INHIBITORY_NT)), -1, 1).astype(np.int8)

    print("reading connectome weights (151M rows) and keeping traced neurons")
    tab = pf.read_table(RAW / FILES["weights"])
    ids = pa.array(body)
    keep = pc.and_(pc.is_in(tab["body_pre"], value_set=ids), pc.is_in(tab["body_post"], value_set=ids))
    tab = tab.filter(keep)
    pre = np.searchsorted(body, tab["body_pre"].to_numpy()).astype(np.int32)
    post = np.searchsorted(body, tab["body_post"].to_numpy()).astype(np.int32)
    cnt = tab["weight"].to_numpy().astype(np.int32)
    del tab
    order = np.lexsort((pre, post))
    pre, post, cnt = pre[order], post[order], cnt[order]
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(post, minlength=n), out=indptr[1:])
    print(f"  {n:,} neurons, {len(cnt):,} connections, {int(cnt.sum()):,} synapses")

    # ---- positions: soma, else the 'toward soma' point, else partners' mean ----
    print("placing neurons")
    pos = np.full((n, 3), np.nan)
    for col in ("somaLocation", "tosomaLocation"):
        have = ann[col].notna().to_numpy() & np.isnan(pos[:, 0])
        if have.any():
            pos[have] = np.stack(ann.loc[have, col].to_numpy()).astype(float)
    direct = ~np.isnan(pos[:, 0])
    both_src, both_dst, both_w = np.concatenate([pre, post]), np.concatenate([post, pre]), np.concatenate([cnt, cnt])
    for _ in range(4):
        missing = np.isnan(pos[:, 0])
        if not missing.any():
            break
        est = _weighted_partner_mean(pos, both_src, both_dst, both_w, n)
        fill = missing & ~np.isnan(est[:, 0])
        pos[fill] = est[fill]
    del both_src, both_dst, both_w
    pos[np.isnan(pos[:, 0])] = np.nanmedian(pos, axis=0)
    rng = np.random.default_rng(0)
    pos[~direct] += rng.normal(0, 500, size=(int((~direct).sum()), 3))  # ~4 um jitter so they don't stack
    print(f"  {int(direct.sum()):,} placed at their soma, {int((~direct).sum()):,} at their partners' mean")

    # Volume axes: x -> fly's left, y -> ventral, z -> posterior (8 nm voxels).
    # Display: x = fly's left (screen right in a frontal view), y = dorsal, z = anterior.
    center = (np.nanmin(pos, 0) + np.nanmax(pos, 0)) / 2
    scale = 2.0 / float((np.nanmax(pos, 0) - np.nanmin(pos, 0)).max())
    disp = np.empty_like(pos)
    disp[:, 0] = (pos[:, 0] - center[0]) * scale
    disp[:, 1] = -(pos[:, 1] - center[1]) * scale
    disp[:, 2] = -(pos[:, 2] - center[2]) * scale
    disp[:, 1] += 0.35  # lift so the brain, not the nerve cord, sits at the orbit centre

    # ---- side ----
    side = ann.somaSide.where(ann.somaSide.isin(["L", "R", "M"]), ann.rootSide.where(ann.rootSide.isin(["L", "R"])))
    midline = float(np.median(pos[ann.somaSide.eq("M").to_numpy(), 0])) if ann.somaSide.eq("M").any() else center[0]
    guess = np.where(pos[:, 0] > midline, "L", "R")
    side = side.fillna(pd.Series(guess, index=ann.index)).to_numpy()

    # ---- visual field from optic-lobe hex columns ----
    # In the lamina (retinotopic, not inverted), hex1+hex2 points dorsal and
    # hex1-hex2 points anterior (checked by regressing L1 soma positions on hex).
    # Neurons without a column get the synapse-weighted mean column of their
    # partners: photoreceptors from their targets, everything else from inputs.
    hex_ = ann[["assignedOlHex1", "assignedOlHex2"]].to_numpy(dtype=float)
    sup = ann.superclass.fillna("").to_numpy()
    rf = hex_.copy()
    sensory = (sup == "ol_sensory")[:, None]
    for _ in range(3):
        need = np.isnan(rf[:, 0])
        r_in = _weighted_partner_mean(rf, pre, post, cnt, n)
        r_out = _weighted_partner_mean(rf, post, pre, cnt, n)
        cand = np.where(sensory | np.isnan(r_in), r_out, r_in)
        fill = need & ~np.isnan(cand[:, 0])
        rf[fill] = cand[fill]
    u = rf[:, 0] - rf[:, 1]
    w = rf[:, 0] + rf[:, 1]
    l1 = (ann.type == "L1").to_numpy() & ~np.isnan(hex_[:, 0])
    ul, wl = hex_[l1, 0] - hex_[l1, 1], hex_[l1, 0] + hex_[l1, 1]
    u_lo, u_hi = np.percentile(ul, [1, 99])
    w_lo, w_hi = np.percentile(wl, [1, 99])
    az_front = (u_hi - u) / (u_hi - u_lo) * 160.0 - 10.0  # 0 = straight ahead, 150 = behind
    elev = ((w - w_lo) / (w_hi - w_lo) * 2 - 1) * 75.0
    # Signed azimuth in degrees, + = fly's right. Left-eye neurons look left.
    az = np.where(side == "L", -az_front, az_front)
    visual = (sup == "ol_sensory") | (sup == "ol_intrinsic") | (sup == "visual_projection")
    az[~visual] = np.nan
    elev[~visual] = np.nan

    # ---- classes / types ----
    sup_to_cls = {s: i for i, (_, _, sups) in enumerate(CLASSES) for s in sups}
    cls = np.array([sup_to_cls.get(s, 0) for s in sup], dtype=np.uint8)
    types = ann.type.fillna(ann.instance).fillna("untyped").to_numpy()
    type_names, type_idx = np.unique(types, return_inverse=True)

    neurons = pd.DataFrame({
        "bodyId": body, "type": types, "instance": ann.instance.fillna(""), "superclass": sup,
        "cls": cls, "side": side, "nt": ntv, "sign": sign, "az": az, "el": elev,
        "x": disp[:, 0], "y": disp[:, 1], "z": disp[:, 2],
        "class": ann["class"].fillna(""), "subclass": ann.subclass.fillna(""),
        "flywireType": ann.flywireType.fillna(""), "synonyms": ann.synonyms.fillna(""),
    })
    neurons.to_parquet(PROCESSED / "neurons.parquet")
    np.savez(
        PROCESSED / "connectome.npz",
        indptr=indptr, indices=pre, counts=cnt.astype(np.uint16), sign=sign,
    )

    # ---- browser payload ----
    side_code = {"L": 1, "R": 2, "M": 3}
    pos32 = disp.astype(np.float32).ravel()
    t16 = type_idx.astype(np.uint16)
    parts, layout, off = [], {}, 0
    for key, arr in (("pos", pos32), ("type", t16), ("cls", cls), ("side", np.array([side_code.get(s, 0) for s in side], np.uint8))):
        b = arr.tobytes()
        layout[key] = off
        parts.append(b)
        off += len(b)
        pad = (-off) % 4
        parts.append(b"\0" * pad)
        off += pad
    (WEB_DATA / "neurons.bin").write_bytes(b"".join(parts))
    counts = np.bincount(cls, minlength=len(CLASSES))
    meta = {
        "n": n,
        "layout": layout,
        "classes": [{"name": nm, "color": col, "count": int(counts[i])} for i, (nm, col, _) in enumerate(CLASSES)],
        "types": type_names.tolist(),
        "dataset": "MaleCNS v1.0",
    }
    (WEB_DATA / "neurons.json").write_text(json.dumps(meta))
    print(f"wrote {PROCESSED} and {WEB_DATA}")


@dataclass
class Connectome:
    dataset: str
    n: int
    indptr: np.ndarray
    indices: np.ndarray
    weights: np.ndarray  # mV, = w_syn * count * sign(pre)
    counts: np.ndarray
    sign: np.ndarray
    neurons: pd.DataFrame

    @property
    def n_connections(self) -> int:
        return int(len(self.indices))

    @property
    def n_synapses(self) -> int:
        return int(self.counts.sum(dtype=np.int64))

    @classmethod
    def load(cls, w_syn: float = 0.275) -> "Connectome":
        f = PROCESSED / "connectome.npz"
        if not f.exists():
            raise SystemExit("No processed connectome yet. Run:  uv run flykart prepare")
        z = np.load(f)
        neurons = pd.read_parquet(PROCESSED / "neurons.parquet")
        counts = z["counts"].astype(np.int32)
        sign = z["sign"].astype(np.int8)
        weights = (w_syn * counts * sign[z["indices"]]).astype(np.float32)
        return cls("MaleCNS v1.0", len(neurons), z["indptr"], z["indices"], weights, counts, sign, neurons)
