"""Where the game meets the connectome: which real neurons each sense drives,
and which neurons' spikes move the kart.

Inputs (Poisson stimulation, like Shiu et al. 2024):
  eyes      ~4,100 photoreceptors (R1-R6, R7, R8), each placed in the visual field
            by the optic-lobe hex column of its synaptic partners
  pursuit   LC10a visual projection neurons (the male chasing pathway), tuned to
            the lead truck's azimuth via their inputs' columns
  looming   LC4 + LPLC2 (the classic giant-fiber inputs), gated by time to contact
  sugar     labellar taste neurons LB3c/d, the taste types that drive MN9 hardest
  touch     antennal Johnston's organ + head / leg bristles, after a crash
  drive     oDN1 (MaleCNS: DNg97), the forward-walking DN Eon also drove directly

Outputs (spike rates, smoothed):
  steer     DNa02 (+DNa01): right > left turns right (Rayshubskiy et al.)
  throttle  oDN1 (DNg97)
  reverse   MDN, the "moonwalker" backward-walking neurons
  jump      DNp01, the giant fiber: any spike launches an escape jump
  feed      MN9, proboscis motor neuron
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .connectome import CLASSES, Connectome

MOTOR_ROWS = [
    {"id": "steer", "label": "Steer", "neurons": "DNa02+DNa01 L|R", "kind": "split", "color": "#ff4fd8", "maxHz": 160},
    {"id": "gas", "label": "Gas", "neurons": "oDN1 (DNg97)", "kind": "single", "color": "#7dff9b", "maxHz": 250},
    {"id": "reverse", "label": "Reverse", "neurons": "MDN moonwalker", "kind": "single", "color": "#ff9a4f", "maxHz": 25},
    {"id": "jump", "label": "Jump", "neurons": "DNp01 giant fiber", "kind": "single", "color": "#5ef1ff", "maxHz": 250},
    {"id": "feed", "label": "Feed", "neurons": "MN9 proboscis", "kind": "single", "color": "#ffcf5a", "maxHz": 160},
]


class Interface:
    def __init__(self, conn: Connectome, device: torch.device):
        self.conn = conn
        self.n = conn.n
        self.device = device
        nz = conn.neurons
        self.T = nz.type.to_numpy()
        self.S = nz.side.to_numpy()
        self.az = nz.az.to_numpy()
        self.el = nz.el.to_numpy()
        self.cls = nz.cls.to_numpy()
        self.pos = nz[["x", "y", "z"]].to_numpy()

        # ---- inputs ----
        photo = np.flatnonzero(nz.type.str.match(r"^R[1-8]").to_numpy() & ~np.isnan(self.az))
        self.photo = photo
        self.lc10a = self.ids("LC10a")
        self.loom = self.ids("LC4", "LPLC2")
        self.sugar = self.ids("LB3c", "LB3d")
        self.touch = self.ids("JO-ED2_b", "BM_vOcci_vPoOr", "SNta02,SNta09")
        self.drive = self.ids("DNg97")
        self.lc10a_sigma = 40.0  # deg; LC10a receptive fields span roughly 20-40 deg
        self.lc10a_peak = 150.0  # Hz, the standard stimulation rate in Shiu et al.
        self.steer_hz = 60.0  # DNa02 rate difference (Hz) that means full lock
        self._prev_eye = None

        # ---- outputs ----
        self.out = {
            "DNa02_L": self.ids("DNa02", side="L"), "DNa02_R": self.ids("DNa02", side="R"),
            "DNa01_L": self.ids("DNa01", side="L"), "DNa01_R": self.ids("DNa01", side="R"),
            "GF": self.ids("DNp01"), "MDN": self.ids("MDN"), "MN9": self.ids("MN9"),
            "oDN1": self.ids("DNg97"), "BDN2": self.ids("DNg100"),
        }
        self.ema = {k: 0.0 for k in self.out}
        self.tau = {"DNa02_L": 0.05, "DNa02_R": 0.05, "DNa01_L": 0.05, "DNa01_R": 0.05,
                    "GF": 0.03, "MDN": 0.12, "MN9": 0.08, "oDN1": 0.15, "BDN2": 0.15}

        # Raster rows: every individual neuron of the output types (+ AOTU019, the
        # LC10a -> DNa02 relay), grouped by type.
        self.key_rows = []
        key_types = [("DNa02", "#ff4fd8"), ("DNa01", "#ff8be6"), ("DNp01", "#5ef1ff"), ("MDN", "#ff9a4f"),
                     ("MN9", "#ffcf5a"), ("DNg97", "#7dff9b"), ("DNg100", "#b7ffc9"), ("AOTU019", "#b48cff")]
        label = {"DNp01": "GF", "DNg97": "oDN1", "DNg100": "BDN2"}
        for t, col in key_types:
            for i in self.ids(t):
                self.key_rows.append({"index": int(i), "group": label.get(t, t), "color": col,
                                      "label": f"{label.get(t, t)} {self.S[i]}"})
        self.key_idx = np.array([r["index"] for r in self.key_rows])

        lateral = np.abs(np.nan_to_num(self.az)) > 45
        self.pokes = {
            "sugar": ("Taste sugar", "LB3c/d → MN9", self.sugar),
            "targetL": ("Target left", "LC10a L → DNa02", self.ids("LC10a", side="L", mask=lateral)),
            "targetR": ("Target right", "LC10a R → DNa02", self.ids("LC10a", side="R", mask=lateral)),
            "loom": ("Looming", "LC4+LPLC2 → GF", self.loom),
            "touch": ("Touch head", "JO + bristles → MDN", self.touch),
            "gf": ("Giant fiber", "DNp01", self.out["GF"]),
            "mdn": ("Moonwalker", "MDN", self.out["MDN"]),
            "odn1": ("Gas", "oDN1", self.drive),
            "eyes": ("Flash eyes", f"{len(photo):,} photoreceptors", photo),
        }

    # ------------------------------------------------------------------ helpers
    def ids(self, *types: str, side: str | None = None, mask: np.ndarray | None = None) -> np.ndarray:
        m = np.isin(self.T, types)
        if side:
            m &= self.S == side
        if mask is not None:
            m &= mask
        return np.flatnonzero(m)

    def poke_group(self, gid: str) -> np.ndarray | None:
        p = self.pokes.get(gid)
        return p[2] if p else None

    def poke_type_of(self, index: int) -> np.ndarray | None:
        if not 0 <= index < self.n:
            return None
        return np.flatnonzero(self.T == self.T[index])

    # ------------------------------------------------------------------ senses -> rates
    def eye_bins(self, n_el: int, n_az: int) -> np.ndarray:
        """Flattened (eye, el, az) bin for every photoreceptor."""
        if getattr(self, "_bins", None) is not None and self._bins_shape == (n_el, n_az):
            return self._bins
        az, el = self.az[self.photo], self.el[self.photo]
        eye = (self.S[self.photo] == "R").astype(int)
        az_grid = [np.linspace(-170, 15, n_az), np.linspace(-15, 170, n_az)]
        el_grid = np.linspace(50, -60, n_el)
        j = np.array([np.abs(az_grid[e] - a).argmin() for e, a in zip(eye, az)])
        i = np.abs(el_grid[None, :] - el[:, None]).argmin(1)
        self._bins = (eye * n_el + i) * n_az + j
        self._bins_shape = (n_el, n_az)
        return self._bins

    def encode(self, senses: dict, settings: dict, pokes: list[dict]) -> torch.Tensor:
        rate = np.zeros(self.n, dtype=np.float32)

        # Eyes. The model (like Shiu et al.) treats every non-GABA/glutamate
        # transmitter as excitatory, so histaminergic photoreceptors *excite* the
        # lamina here, where in the fly light inhibits it. Driving photoreceptors
        # with darkness and OFF edges makes lamina activity follow darkness, as the
        # real L1/L2 cells do.
        eye = senses["eye"]
        flat = eye.ravel()
        off = np.zeros_like(flat) if self._prev_eye is None else np.clip(self._prev_eye - flat, 0, 1)
        self._prev_eye = flat.copy()
        n_el, n_az = eye.shape[1], eye.shape[2]
        bins = self.eye_bins(n_el, n_az)
        rate[self.photo] = np.minimum(3.0 + 15.0 * (1.0 - flat[bins]) + 700.0 * off[bins], 300.0)

        # Pursuit: LC10a neurons whose column sits near the truck's azimuth.
        tgt = senses.get("target")
        if tgt is not None:
            az_t, strength = tgt
            w = np.exp(-0.5 * ((self.az[self.lc10a] - az_t) / self.lc10a_sigma) ** 2)
            rate[self.lc10a] += self.lc10a_peak * strength * w

        # Looming: LC4 + LPLC2 around each approaching obstacle's azimuth.
        for az_o, drive in senses.get("loom", []):
            w = np.exp(-0.5 * ((np.nan_to_num(self.az[self.loom]) - az_o) / 45.0) ** 2)
            rate[self.loom] = np.maximum(rate[self.loom], 150.0 * drive * w)

        if senses.get("sugar", 0) > 0:
            rate[self.sugar] += 150.0 * senses["sugar"]
        if senses.get("touch", 0) > 0:
            rate[self.touch] += 150.0 * senses["touch"]
        # oDN1 sits under strong feedback inhibition (VES104, DNge054, and AOTU019 from
        # the pursuit pathway), which crushes weak drive; 0-300 Hz breaks through.
        rate[self.drive] += 300.0 * float(settings.get("drive", 0.0))

        for p in pokes:
            rate[p["idx"]] = np.maximum(rate[p["idx"]], 150.0)
        return torch.from_numpy(rate)

    # ------------------------------------------------------------------ spikes -> kart
    def decode(self, counts: np.ndarray, spk_idx: np.ndarray, frame_s: float):
        hz = {}
        for k, idx in self.out.items():
            r = counts[idx].sum() / (len(idx) * frame_s) if len(idx) else 0.0
            a = 1.0 - math.exp(-frame_s / self.tau[k])
            self.ema[k] += (r - self.ema[k]) * a
            hz[k] = r
        e = self.ema
        left = e["DNa02_L"] + 0.5 * e["DNa01_L"]
        right = e["DNa02_R"] + 0.5 * e["DNa01_R"]
        act = {
            "steer": float(np.clip((right - left) / self.steer_hz, -1.0, 1.0)),
            "throttle": float(np.clip((e["oDN1"] + 0.5 * e["BDN2"]) / 180.0, 0.0, 1.0)),
            "reverse": bool(e["MDN"] > 5.0),
            "jump": bool(counts[self.out["GF"]].sum() > 0),
            "feed": float(np.clip(e["MN9"] / 100.0, 0.0, 1.0)),
        }
        motor = {
            "steer": [round(left, 1), round(right, 1)],
            "gas": round(e["oDN1"], 1),
            "reverse": round(e["MDN"], 1),
            "jump": round(e["GF"], 1),
            "feed": round(e["MN9"], 1),
        }
        key_spikes = counts[self.key_idx].astype(int).tolist()
        class_counts = np.bincount(self.cls[spk_idx], weights=counts[spk_idx], minlength=len(CLASSES)).astype(int).tolist()
        return act, motor, key_spikes, class_counts

    def reset(self) -> None:
        self.ema = {k: 0.0 for k in self.out}
        self._prev_eye = None

    # ------------------------------------------------------------------ browser
    def init_payload(self) -> dict:
        def centroid_group(name, idx, color, label):
            return {"name": name, "indices": [int(i) for i in idx], "color": color, "label": label}

        groups = [
            centroid_group("DNa02_L", self.out["DNa02_L"], "#ff4fd8", "DNa02 L"),
            centroid_group("DNa02_R", self.out["DNa02_R"], "#ff4fd8", "DNa02 R"),
            centroid_group("GF", self.out["GF"], "#5ef1ff", "Giant fiber"),
            centroid_group("MDN", self.out["MDN"], "#ff9a4f", "Moonwalker"),
            centroid_group("MN9", self.out["MN9"], "#ffcf5a", "MN9 feed"),
            centroid_group("oDN1", self.out["oDN1"], "#7dff9b", "oDN1 gas"),
            centroid_group("LC10a_L", self.ids("LC10a", side="L"), "#5ef1ff", "LC10a L"),
            centroid_group("LC10a_R", self.ids("LC10a", side="R"), "#5ef1ff", "LC10a R"),
            centroid_group("sugar", self.sugar, "#ffcf5a", "Sugar taste"),
            {"name": "DNa01", "indices": [int(i) for i in np.r_[self.out["DNa01_L"], self.out["DNa01_R"]]], "color": "#ff8be6", "label": ""},
            {"name": "BDN2", "indices": [int(i) for i in self.out["BDN2"]], "color": "#b7ffc9", "label": ""},
        ]
        pokes = [{"id": k, "label": v[0], "sub": v[1]} for k, v in self.pokes.items()]
        return {"groups": groups, "motorRows": MOTOR_ROWS, "keyRows": self.key_rows, "pokes": pokes}
