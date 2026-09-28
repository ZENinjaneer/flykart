# Artist models

Drop `.glb` files in this folder and list them in `models.json`. When the
dashboard loads, each one replaces a built-in model or is placed as scenery. A
missing or broken file just falls back to the built-in shapes.

```json
{
  "models": [
    { "file": "kart.glb",   "role": "kart",     "size": 3,  "rotate": [0, 90, 0], "credit": "Kart by …" },
    { "file": "truck.glb",  "role": "truck",    "size": 4 },
    { "file": "rock.glb",   "role": "obstacle", "size": 2.4 },
    { "file": "candy.glb",  "role": "sugar",    "size": 1.2 },
    { "file": "temple.glb", "role": "scenery",  "size": 40, "place": [[-20, -5, 30]] }
  ]
}
```

| Field | Meaning |
|---|---|
| `file` | The `.glb` (or `.gltf`) in this folder |
| `role` | `kart`, `truck`, `obstacle`, `sugar`, or `scenery` |
| `size` | Scale so the widest horizontal side is this many world units. For reference, the kart is ~2.6 and the track is 10 wide. Use `scale` instead for a raw multiplier. |
| `rotate` | `[x, y, z]` degrees applied first. Karts and trucks should face **+X**. |
| `lift` | Raise the model off the ground (world units) |
| `place` | Scenery only: a list of `[x, y, headingDegrees]` spots on the ground. The track spans roughly x −60…60, y −45…35. |
| `credit` | Shown in the corner of the kart view |

Models are centred and stood on the ground automatically, and animations
play on a loop. Keep each file under ~20 MB (2K textures) so the repo and
the browser stay fast.

Anything added here needs a license that allows it in this public repo.
Record the artist's terms in `LICENSE-ARTWORK.md` in this folder. The code's
MIT license does not cover artwork.
