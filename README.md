# infinity-designer

A browser sandbox for designing infinity mirrors and LED light art. Put LED strips and
mirrors (fully reflective or half-silvered) inside a box, look at it from a viewer's eye,
and see the real thing: every virtual image, dimmed honestly by every bounce.

Static site, no backend. `npm run build` emits `dist/`.

```
npm install
npm run dev      # http://localhost:5173
npm run build    # -> dist/
```

## How it works

**The picture is path traced, on the GPU, every frame.** A WebGL2 fragment shader shoots
one ray per pixel from the eye and walks it through the scene. At a mirror it flips a coin:
reflect with probability `reflect`, pass straight through with probability `transmit`,
otherwise die. At a matte wall it continues with probability `albedo` in a random
direction. Because each event is sampled with exactly its own probability, the ray never
carries a weight; it just reports the emitter it eventually hits. Frames accumulate into a
float buffer and the display shows the running mean, so the image sharpens while you
watch and resets when you move. A 4070 does about 3 ms per sample at 1360×820 with 64
bounces.

**Half-silvered glass is the coin flip.** The classic infinity mirror is a back mirror
(`reflect 0.95`), a front pane of 50/50 film (`reflect 0.5, transmit 0.45`), and a strip
of LEDs between them. The k-th image the eye sees has gone through the front once and
bounced back and forth k times, so its brightness is `T · (R_back · R_front)^k`. That
falloff comes out of the sim on its own; nothing is hand-tuned.

**LED strips are capsules with a pitch.** A strip is a segment with a radius and a
distance between emitters (60/m is 16.67 mm). The body is opaque; a hit within one radius
of an LED centre emits, otherwise it is dark PCB. Real strips are discrete dots and so are
these.

**Why ray tracing, not method of images.** Virtual images tile cleanly only when the
mirrors form a reflection group (parallel pair, kaleidoscope prism). One tilted mirror in
a box breaks the tiling. A handful of rectangles and capsules is cheap to trace directly,
and the tilted case falls out of the same loop.

## Designing

The shell is any convex polyhedron: pick one from the library (Platonic solids, prisms,
pyramids, antiprisms) or give a custom vertex list, then size, stretch and rotate it. Every
face is open, matte or mirror. Click a face, edge, cut outline or strip in the view to
select it and edit it in the panel.

- **Faces** get a material. "All faces like this" makes it the default.
- **Edges** get LED strips, inset into the solid along the bisector of the two faces they
  join, so "LEDs on all edges" of a dodecahedron is the classic infinity dodecahedron.
- **Cuts** are planes through the shell, clipped to its outline, that become interior
  mirrors: the fold-space move. Offset, yaw, pitch and scale are live sliders. The classic
  infinity mirror is a flat box with one 50/50 cut just behind the open front face.
- **Free strips** are segments anywhere, for emitters that don't sit on an edge.

The scene JSON under the panel is the escape hatch and the file format; `save json` and
`open` round-trip it.

## Scene file

Units are millimetres, the shell is centred at the origin, `+z` faces the viewer, `+y` is up.

```jsonc
{
  "shell": {
    "poly": "cube",                      // or "custom" with "vertices": [[x,y,z], ...]
    "size": 150,                         // half-width: the unit solid spans ±1 on its largest axis
    "stretch": [1, 1, 0.2], "rotate": [0, 0, 0],
    "defaultFace": { "kind": "matte", "albedo": 0.04 },
    "faces": { "3": { "kind": "open" }, "2": { "kind": "mirror", "reflect": 0.95, "transmit": 0 } }
  },
  "cuts": [ { "normal": [0, 0, 1], "offset": 28, "scale": 1,
              "material": { "kind": "mirror", "reflect": 0.5, "transmit": 0.45 } } ],
  "edgeStrips": [ { "edge": 0 }, { "edge": 3, "color": [1, 0.1, 0.1] } ],
  "edgeDefaults": { "color": [1, 0.75, 0.45], "radiance": 30, "pitch": 16.67, "radius": 1.5, "inset": 12 },
  "strips": [ { "a": [-80, 0, -25], "b": [80, 0, -25], "color": [0.2, 0.9, 1],
                "radiance": 25, "pitch": 16.67, "radius": 1.5 } ],
  "eye": { "pos": [120, 80, 700], "target": [0, 0, 0], "fov": 32 }
}
```

Face and edge indices come from the hull, which is deterministic for a given vertex set;
the face dropdown in the panel lists them.

## Status

Renderer, polyhedral shells, cutting-plane mirrors, edge and free LED strips, click-to-select
editor. See `BRIEF.md` for scope and what comes next.
