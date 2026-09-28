"""The kart world: a closed track, the fly's kart, a lead car to chase,
obstacles to jump, and sugar cubes to eat. Pure numpy, no brain in here.

Each tick the world reports what the fly *senses* (abstract channels) and
accepts what the fly *does* (steer / throttle / reverse / jump / feed).
interface.py maps those channels onto real neurons.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

TRACK_HALF_WIDTH = 5.0
KART_RADIUS = 1.1
OBSTACLE_RADIUS = 1.2
SUGAR_RADIUS = 1.6
LEAD_RADIUS = 1.4
EYE_HEIGHT = 1.2


def _catmull_rom_loop(pts: np.ndarray, samples_per_seg: int = 40) -> np.ndarray:
    out = []
    n = len(pts)
    for i in range(n):
        p0, p1, p2, p3 = pts[(i - 1) % n], pts[i], pts[(i + 1) % n], pts[(i + 2) % n]
        for t in np.linspace(0.0, 1.0, samples_per_seg, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(
                0.5
                * (
                    2 * p1
                    + (-p0 + p2) * t
                    + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                    + (-p0 + 3 * p1 - 3 * p2 + p3) * t3
                )
            )
    return np.asarray(out)


class Track:
    """Closed centerline polyline with arclength lookup."""

    CONTROL = np.array(
        [
            [0, -40], [30, -42], [52, -28], [56, 0], [44, 22], [20, 26],
            [6, 12], [-10, 8], [-24, 26], [-46, 30], [-58, 8], [-50, -24], [-28, -38],
        ],
        dtype=float,
    )

    def __init__(self):
        self.pts = _catmull_rom_loop(self.CONTROL)
        seg = np.diff(np.vstack([self.pts, self.pts[:1]]), axis=0)
        self.seg_len = np.linalg.norm(seg, axis=1)
        self.cum = np.concatenate([[0.0], np.cumsum(self.seg_len)])
        self.length = float(self.cum[-1])
        self.tangent = seg / self.seg_len[:, None]
        self._build_mask()

    MASK_RES = 0.5  # world units per cell

    def _build_mask(self) -> None:
        """Rasterise 'on track' once so the fly's eyes can look it up per ray."""
        lo = self.pts.min(0) - TRACK_HALF_WIDTH - 2
        hi = self.pts.max(0) + TRACK_HALF_WIDTH + 2
        xs = np.arange(lo[0], hi[0], self.MASK_RES)
        ys = np.arange(lo[1], hi[1], self.MASK_RES)
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        cells = np.stack([gx.ravel(), gy.ravel()], 1)
        on = np.zeros(len(cells), bool)
        for i in range(0, len(cells), 20000):
            on[i:i + 20000] = self._on_track_exact(cells[i:i + 20000])
        self._mask = on.reshape(gx.shape)
        self._mask_lo = lo

    def point_at(self, s: float) -> tuple[np.ndarray, float]:
        """Position and heading at arclength s (wraps)."""
        s = s % self.length
        i = int(np.searchsorted(self.cum, s, side="right") - 1)
        i = min(i, len(self.pts) - 1)
        f = (s - self.cum[i]) / self.seg_len[i]
        p = self.pts[i] + f * (self.pts[(i + 1) % len(self.pts)] - self.pts[i])
        t = self.tangent[i]
        return p, math.atan2(t[1], t[0])

    def nearest(self, xy: np.ndarray) -> tuple[float, float, float]:
        """(distance to centerline, arclength of nearest point, track heading there)."""
        a = self.pts
        b = np.roll(self.pts, -1, axis=0)
        ab = b - a
        t = np.clip(((xy - a) * ab).sum(1) / (ab * ab).sum(1), 0.0, 1.0)
        proj = a + t[:, None] * ab
        d = np.linalg.norm(proj - xy, axis=1)
        i = int(np.argmin(d))
        tan = self.tangent[i]
        return float(d[i]), float(self.cum[i] + t[i] * self.seg_len[i]), math.atan2(tan[1], tan[0])

    def on_track_mask(self, xy: np.ndarray) -> np.ndarray:
        """On-track test for many points (M x 2), via the precomputed grid."""
        ij = np.floor((xy - self._mask_lo) / self.MASK_RES).astype(int)
        ok = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < self._mask.shape[0]) & (ij[:, 1] < self._mask.shape[1])
        out = np.zeros(len(xy), bool)
        out[ok] = self._mask[ij[ok, 0], ij[ok, 1]]
        return out

    def _on_track_exact(self, xy: np.ndarray) -> np.ndarray:
        a = self.pts[None, :, :]
        ab = (np.roll(self.pts, -1, axis=0) - self.pts)[None, :, :]
        ap = xy[:, None, :] - a
        t = np.clip((ap * ab).sum(2) / (ab * ab).sum(2), 0.0, 1.0)
        d = np.linalg.norm(ap - t[..., None] * ab, axis=2).min(1)
        return d < TRACK_HALF_WIDTH


@dataclass
class Kart:
    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0
    speed: float = 0.0
    z: float = 0.0  # jump height
    vz: float = 0.0
    steer: float = 0.0
    throttle: float = 0.0
    reversing: bool = False
    feeding: float = 0.0  # proboscis extension 0..1
    stunned: float = 0.0  # seconds left after a crash


@dataclass
class Obstacle:
    s: float  # arclength position
    offset: float  # lateral offset from centerline
    x: float = 0.0
    y: float = 0.0
    hit: bool = False
    cleared: bool = False
    id: int = 0
    t_done: float = 0.0


@dataclass
class Sugar:
    x: float
    y: float
    eaten_at: float = -1e9
    id: int = 0


@dataclass
class Events:
    crashes: int = 0
    jumps: int = 0
    cleared: int = 0
    sugar: int = 0
    laps: float = 0.0
    log: list = field(default_factory=list)


class World:
    MAX_SPEED = 14.0
    MAX_REVERSE = 4.0
    ACCEL = 10.0
    DRAG = 0.9
    TURN_RATE = 2.2  # rad/s at full steer

    def __init__(self, seed: int = 1, mode: str = "chase"):
        self.rng = np.random.default_rng(seed)
        self.track = Track()
        self.mode = mode
        self.t = 0.0
        self.reset()

    # ------------------------------------------------------------------ setup
    def reset(self) -> None:
        self.t = 0.0
        p, h = self.track.point_at(0.0)
        self.kart = Kart(x=float(p[0]), y=float(p[1]), heading=h)
        self.lead_s = 12.0
        self.lead_speed = 0.0
        self.events = Events()
        self.on_track_frames = 0
        self.frames = 0
        self._prev_ang = {}
        self._prev_eye = None
        self._last_s = 0.0
        self._next_id = 0
        self.obstacles: list[Obstacle] = []
        self.sugars: list[Sugar] = []
        n_obs = 10
        for i in range(n_obs):
            s = self.track.length * (i + 0.6) / n_obs
            self._add_obstacle(s, self.rng.uniform(-2.0, 2.0))
        for i in range(16):
            s = self.track.length * (i + 0.25) / 16
            p, h = self.track.point_at(s)
            off = self.rng.uniform(-3.0, 3.0)
            nx, ny = -math.sin(h), math.cos(h)
            self.sugars.append(Sugar(float(p[0] + nx * off), float(p[1] + ny * off), id=i))

    def _add_obstacle(self, s: float, offset: float) -> None:
        p, h = self.track.point_at(s)
        nx, ny = -math.sin(h), math.cos(h)
        o = Obstacle(s=s, offset=offset, x=float(p[0] + nx * offset), y=float(p[1] + ny * offset), id=self._next_id)
        self._next_id += 1
        self.obstacles.append(o)

    # ------------------------------------------------------------------ geometry
    def lead_pose(self) -> tuple[np.ndarray, float]:
        return self.track.point_at(self.lead_s)

    def _egocentric(self, x: float, y: float) -> tuple[float, float]:
        """(distance, azimuth) of a point; azimuth > 0 means to the fly's RIGHT."""
        k = self.kart
        dx, dy = x - k.x, y - k.y
        d = math.hypot(dx, dy)
        ang = math.atan2(dy, dx) - k.heading
        ang = (ang + math.pi) % (2 * math.pi) - math.pi
        return d, -ang  # world is counter-clockwise positive; flip so + = right

    # ------------------------------------------------------------------ senses
    def sense(self, dt: float) -> dict:
        """What the fly perceives right now (unitless drives, azimuths in degrees, + = right)."""
        k = self.kart
        out: dict = {"target": None, "loom": []}

        # Lead truck -> pursuit channel (LC10a). Strength grows with apparent size.
        target_l = target_r = 0.0
        if self.mode == "chase":
            lp, _ = self.lead_pose()
            d, az = self._egocentric(float(lp[0]), float(lp[1]))
            if d > 0.1 and abs(az) < math.radians(160):
                size = 2 * math.atan(LEAD_RADIUS / d)
                strength = float(np.clip(0.45 + size / math.radians(25), 0.0, 1.0))
                out["target"] = (math.degrees(az), strength)
                target_r = strength * _sigmoid((az + math.radians(10)) / math.radians(8))
                target_l = strength * _sigmoid((-az + math.radians(10)) / math.radians(8))
        out["target_l"], out["target_r"] = target_l, target_r

        # Obstacles on a collision course -> looming channel (LC4 / LPLC2). The
        # giant fiber fires when a looming object gets big fast; we model that
        # read-out by gating on time to contact (tau = distance / closing speed).
        loom_l = loom_r = 0.0
        closing_speed = max(k.speed, 0.0)
        for o in self.obstacles:
            if o.hit or o.cleared or closing_speed < 0.8:
                continue
            d, az = self._egocentric(o.x, o.y)
            if d > 30 or abs(az) > math.radians(80):
                continue
            ahead = d * math.cos(az)
            miss = abs(d * math.sin(az))
            if ahead <= 0 or miss > OBSTACLE_RADIUS + KART_RADIUS + 0.6:
                continue
            tau = ahead / closing_speed
            drive = float(np.clip((0.65 - tau) / 0.35, 0.0, 1.0))
            if drive <= 0:
                continue
            out["loom"].append((math.degrees(az), drive))
            if az >= 0:
                loom_r = max(loom_r, drive)
            if az <= 0:
                loom_l = max(loom_l, drive)
        out["loom_l"], out["loom_r"] = loom_l, loom_r

        # Taste: currently on a sugar cube?
        out["sugar"] = 1.0 if any(self.t - s.eaten_at < 0.4 for s in self.sugars) else 0.0
        # Touch: just bumped into something.
        out["touch"] = 1.0 if k.stunned > 0.2 else 0.0
        out["eye"] = self._render_eyes()
        return out

    EYE_AZ = 20  # azimuth bins per eye
    EYE_EL = 8  # elevation bins

    def _render_eyes(self) -> np.ndarray:
        """Brightness image per eye, shape (2, EYE_EL, EYE_AZ). Eye 0 = left."""
        k = self.kart
        az_l = np.linspace(math.radians(-170), math.radians(15), self.EYE_AZ)
        az_r = np.linspace(math.radians(-15), math.radians(170), self.EYE_AZ)
        el = np.linspace(math.radians(50), math.radians(-60), self.EYE_EL)
        img = np.zeros((2, self.EYE_EL, self.EYE_AZ))
        for e, az in enumerate((az_l, az_r)):
            A, E = np.meshgrid(az, el)
            sky = E > 0
            # Ground intersection for rays below the horizon.
            dist = np.where(sky, np.inf, (EYE_HEIGHT + k.z) / np.tan(np.maximum(-E, 1e-3)))
            world_ang = k.heading - A  # az>0 is right = clockwise
            gx = k.x + np.cos(world_ang) * np.minimum(dist, 200)
            gy = k.y + np.sin(world_ang) * np.minimum(dist, 200)
            on = self.track.on_track_mask(np.stack([gx.ravel(), gy.ravel()], 1)).reshape(A.shape)
            b = np.where(sky, 0.9 - 0.3 * E, np.where(on, 0.25, 0.5))
            b = np.where(~sky & (dist > 120), 0.55, b)
            # Objects paint dark (obstacles) or bright (lead car, sugar) blobs.
            objs = [(o.x, o.y, OBSTACLE_RADIUS, 1.4, 0.05) for o in self.obstacles if not o.hit and not o.cleared]
            if self.mode == "chase":
                lp, _ = self.lead_pose()
                objs.append((float(lp[0]), float(lp[1]), LEAD_RADIUS, 1.3, 1.0))
            for ox, oy, rad, hgt, val in objs:
                d, oaz = self._egocentric(ox, oy)
                if d > 80 or d < 0.2:
                    continue
                half_w = math.atan(rad / d)
                top = math.atan((hgt - EYE_HEIGHT - k.z) / d)
                bot = math.atan((-EYE_HEIGHT - k.z) / d)
                m = (np.abs(_wrap(A - oaz)) < half_w) & (E < top + 0.02) & (E > bot)
                b = np.where(m, val, b)
            img[e] = b
        return img

    # ------------------------------------------------------------------ act
    def step(self, dt: float, act: dict) -> None:
        """Advance physics. act: steer [-1,1] (+ = right), throttle [0,1],
        reverse bool, jump bool, feed [0,1], assist [0,1] (autopilot blend)."""
        k = self.kart
        self.t += dt

        # The truck teases: it keeps ~12 units ahead of the kart (straight-line), so a
        # fly that cuts corners or stalls doesn't lose it for good.
        if self.mode == "chase":
            lp, _ = self.lead_pose()
            dist = math.hypot(lp[0] - k.x, lp[1] - k.y)
            want = max(k.speed, 0.0) + 0.35 * (12.0 - dist)
            self.lead_speed += (np.clip(want, 0.0, 10.0) - self.lead_speed) * min(1.0, dt * 2.0)
            self.lead_s = (self.lead_s + self.lead_speed * dt) % self.track.length

        steer = float(np.clip(act.get("steer", 0.0), -1, 1))
        assist = float(np.clip(act.get("assist", 0.0), 0, 1))
        if assist > 0:
            steer = (1 - assist) * steer + assist * self._autopilot_steer()
        throttle = float(np.clip(act.get("throttle", 0.0), 0, 1))
        k.steer += (steer - k.steer) * min(1.0, dt * 12)
        k.throttle = throttle
        k.reversing = bool(act.get("reverse", False))
        k.feeding += (float(act.get("feed", 0.0)) - k.feeding) * min(1.0, dt * 10)

        if act.get("jump") and k.z <= 0.0:
            k.vz = 10.0
            self.events.jumps += 1
            self.events.log.append((self.t, "jump"))

        d_center, s_here, _ = self.track.nearest(np.array([k.x, k.y]))
        off_track = d_center > TRACK_HALF_WIDTH
        self.frames += 1
        self.on_track_frames += 0 if off_track else 1
        top = self.MAX_SPEED * (0.85 if off_track else 1.0)  # flies don't care much about roads
        if k.stunned > 0:
            k.stunned -= dt
            accel = -8.0 * math.copysign(1, k.speed) if abs(k.speed) > 0.3 else 0.0
        elif k.reversing:
            accel = -self.ACCEL * 0.8 if k.speed > -self.MAX_REVERSE else 0.0
        else:
            accel = self.ACCEL * throttle
        k.speed += (accel - self.DRAG * k.speed - (0.25 * k.speed if off_track else 0.0)) * dt
        k.speed = float(np.clip(k.speed, -self.MAX_REVERSE, top))
        grip = 1.0 if k.z <= 0 else 0.3
        k.heading -= self.TURN_RATE * k.steer * grip * dt * np.clip(abs(k.speed) / 3.0, 0.15, 1.0) * np.sign(k.speed or 1)
        k.x += math.cos(k.heading) * k.speed * dt
        k.y += math.sin(k.heading) * k.speed * dt
        if k.z > 0 or k.vz > 0:
            k.vz -= 22.0 * dt
            k.z = max(0.0, k.z + k.vz * dt)
            if k.z == 0.0:
                k.vz = 0.0

        # Keep the kart in the arena.
        r = math.hypot(k.x, k.y)
        if r > 90:
            k.x, k.y = k.x * 90 / r, k.y * 90 / r
            k.speed *= 0.3

        # Collisions / pickups.
        for o in self.obstacles:
            if o.hit or o.cleared:
                continue
            d = math.hypot(o.x - k.x, o.y - k.y)
            if d < OBSTACLE_RADIUS + KART_RADIUS:
                o.t_done = self.t
                if k.z > 0.9:
                    o.cleared = True
                    self.events.cleared += 1
                    self.events.log.append((self.t, "cleared"))
                else:
                    o.hit = True
                    k.stunned = 1.0
                    k.speed *= 0.2
                    self.events.crashes += 1
                    self.events.log.append((self.t, "crash"))
        for s in self.sugars:
            if self.t - s.eaten_at > 12.0 and math.hypot(s.x - k.x, s.y - k.y) < SUGAR_RADIUS + KART_RADIUS:
                s.eaten_at = self.t
                self.events.sugar += 1
                self.events.log.append((self.t, "sugar"))

        # Respawn obstacles a few seconds later, once the kart is well away from them.
        for o in self.obstacles:
            if (o.hit or o.cleared) and self.t - o.t_done > 4.0 and math.hypot(o.x - k.x, o.y - k.y) > 25:
                p, h = self.track.point_at(o.s)
                o.offset = float(self.rng.uniform(-2.0, 2.0))
                nx, ny = -math.sin(h), math.cos(h)
                o.x, o.y = float(p[0] + nx * o.offset), float(p[1] + ny * o.offset)
                o.hit = o.cleared = False
                self._prev_ang.pop(o.id, None)

        ds = (s_here - self._last_s + self.track.length / 2) % self.track.length - self.track.length / 2
        if abs(ds) < 1.0:  # ignore jumps of the nearest point while cutting across the infield
            self.events.laps += ds / self.track.length
        self._last_s = s_here
        self.events.log = self.events.log[-20:]

    def _autopilot_steer(self) -> float:
        """Pure-pursuit on the track centerline (the 'training wheels')."""
        k = self.kart
        _, s, _ = self.track.nearest(np.array([k.x, k.y]))
        p, _ = self.track.point_at(s + 9.0)
        _, az = self._egocentric(float(p[0]), float(p[1]))
        return float(np.clip(az * 1.8, -1, 1))

    # ------------------------------------------------------------------ serialise
    def static_state(self) -> dict:
        return {
            "track": np.round(self.track.pts, 3).tolist(),
            "halfWidth": TRACK_HALF_WIDTH,
            "eye": {"az": self.EYE_AZ, "el": self.EYE_EL},
        }

    def dynamic_state(self) -> dict:
        k = self.kart
        lp, lh = self.lead_pose()
        return {
            "t": round(self.t, 3),
            "kart": {
                "x": round(k.x, 3), "y": round(k.y, 3), "h": round(k.heading, 4), "v": round(k.speed, 3),
                "z": round(k.z, 3), "steer": round(k.steer, 3), "throttle": round(k.throttle, 3),
                "rev": k.reversing, "feed": round(k.feeding, 3), "stun": round(max(0.0, k.stunned), 2),
            },
            "lead": {"x": round(float(lp[0]), 3), "y": round(float(lp[1]), 3), "h": round(lh, 4)} if self.mode == "chase" else None,
            "obstacles": [
                {"id": o.id, "x": round(o.x, 2), "y": round(o.y, 2), "hit": o.hit, "ok": o.cleared}
                for o in self.obstacles
            ],
            "sugars": [
                {"id": s.id, "x": round(s.x, 2), "y": round(s.y, 2), "on": self.t - s.eaten_at > 12.0}
                for s in self.sugars
            ],
            "events": {
                "crashes": self.events.crashes, "jumps": self.events.jumps, "cleared": self.events.cleared,
                "sugar": self.events.sugar, "laps": round(self.events.laps, 3),
                "log": [(round(t, 2), e) for t, e in self.events.log[-6:]],
            },
        }


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi
