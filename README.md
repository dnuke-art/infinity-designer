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
these. Each LED's colour comes from a texture that a tiny pattern shader fills every frame,
so animating the pixels never touches the geometry.

**Tinted and animated mirrors are the same coin flip.** Reflect and transmit are RGB. The
path samples the event by the mean probability and carries the per-channel ratio in a
colour throughput, so a bluish film shifts every deeper image bluer for free. Animated
materials swap the static values for a texture lookup at the hit point.

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
  Density is set as LEDs per metre or pitch in mm. A `cone` is the strip's viewing angle
  (the datasheet's full angle at half intensity; 120° is a bare SMD strip; the default is
  omnidirectional): emission falls off as a Gaussian in angle and fades out just behind
  the package plane, so a wall-mounted strip still feeds the tunnel at ~20% of its peak.
  `shadeFace` picks which face carries the lip, or 'both' for an opaque frame band on
  both faces with the strip shining inward from the corner; left unset, the more
  transparent face gets it, or both when the faces are equally see-through (an infinity
  dodecahedron). A `shade` is an
  opaque lip of that width along the edge, lying in the viewing face and reaching inward,
  which hides the strip from the front so only its reflections show; the strip then mounts
  on the other face and points along its inward normal. The "shaded infinity mirror"
  preset is the classic box with lips and 120° strips.
- **Cuts** are planes through the shell, clipped to its outline, that become interior
  mirrors: the fold-space move. Offset, yaw, pitch and scale are live sliders. The classic
  infinity mirror is a flat box with one 50/50 cut just behind the open front face.
- **Free strips** are segments anywhere, for emitters that don't sit on an edge.
- **Patterns** drive the LEDs as addressable pixels. Every LED has a global index in wiring
  order (strips in scene order, each from its data-in end; `reverse` flips one). A pattern
  is a GLSL function `vec3 pattern(Led led, float t)` evaluated on the GPU once per LED
  with its index, count, strip number, position along the strip, world position and base
  colour. Pick a preset (rainbow, chase, plane wave, sparkle …) or write your own in the
  panel; compile errors show inline. Play animates it: while playing, the accumulator
  becomes a moving average over the last dozen frames so the preview stays smooth, and
  pausing lets it converge fully. "Export LED map" writes the LED positions in index order
  as a JSON array, the pixel-map format Pixelblaze and similar controllers take.
- **Animated materials and screens.** A mirror or screen face or cut can carry a GLSL
  program `void material(Surf s, float t, inout vec3 R, inout vec3 T, inout vec3 E)` that
  sets per-channel reflect, transmit and emitted radiance as a function of position on the
  surface and time. Uniform in position it is a shutter or a switchable mirror; varying, it
  is a pixel mask on a transparent LCD; with E it is an OLED or backlit LCD showing
  content. Programs run on the GPU into three 128×128 tiles per material each frame, and
  a picture you load is available to them as `uImage`. Presets cover a switchable front
  mirror, a ring mask, an OLED back panel, an LCD behind a back half mirror, a transparent
  LCD as a colour filter over the back mirror, and a half-backlit transparent LCD.
- **Stateful patterns.** A pattern can read what any LED was last frame (`prevState`,
  `prevColor`) and set `state` to carry a value forward, so simulations run on the strip.
  The "fire (Fire2012)" preset is Kriegsman's classic: heat cools, drifts up each strip from
  its data-in end and sparks at the base. Reverse an edge to choose which way its flames go.
- **Fog.** A homogeneous scattering medium fills the shell (density per metre, albedo,
  Henyey-Greenstein g). The camera path may scatter before reaching a surface, and at the
  scatter point the fog is lit by next-event estimation: a shadow ray aims at an LED or at
  one of its virtual images, and validates the image by actually bouncing off the listed
  mirrors in order (up to 1200 images, 4 mirrors deep). Deeper images still arrive by
  chance. Beams and fan sheets glow where a camera ray crosses them.
- **Lasers.** A laser is a point, a direction, a colour, a power and a fan angle (0 = a
  beam, otherwise a line-generator fan with a roll about the beam). Rays are followed
  through the mirrors on the CPU, splitting at half-silvered ones, and leave emissive lines
  where they land on matte surfaces and glowing sheets in the fog. Click a beam to select
  the laser. Animated mirrors reflect lasers with their static values.
- **STL shells.** "Import STL as the shell" loads any closed triangle mesh, concave or not;
  the triangles become the faces, every mesh edge can carry LEDs, and cuts clip against the
  mesh's convex hull. The heart from hvaw-show ships as the "mirror heart" preset.

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
              "material": { "kind": "mirror", "reflect": 0.5, "transmit": 0.45 } },
            { "normal": [0, 0, 1], "offset": -24,
              "material": { "kind": "screen", "reflect": 0.05, "transmit": 0.4, "emit": [0, 0, 0],
                            "program": "void material(Surf s, float t, inout vec3 R, inout vec3 T, inout vec3 E) { E = vec3(2.0) * step(0.0, s.uv.x); }" } } ],
  "edgeStrips": [ { "edge": 0 }, { "edge": 3, "color": [1, 0.1, 0.1], "reverse": true } ],
  "edgeDefaults": { "color": [1, 0.75, 0.45], "radiance": 30, "pitch": 16.67, "radius": 1.5, "inset": 12 },
  "strips": [ { "a": [-80, 0, -25], "b": [80, 0, -25], "color": [0.2, 0.9, 1],
                "radiance": 25, "pitch": 16.67, "radius": 1.5 } ],
  "pattern": { "name": "custom", "source": "vec3 pattern(Led led, float t) { return led.color; }" },
  "fog": { "density": 4, "albedo": [0.9, 0.9, 0.9], "g": 0.5 },
  "lasers": [ { "pos": [-140, -130, 20], "dir": [1, 0.35, -0.6], "color": [0.2, 1, 0.3], "power": 6, "fan": 50, "roll": 10, "radius": 1, "rays": 20 } ],
  "eye": { "pos": [120, 80, 700], "target": [0, 0, 0], "fov": 32, "exposure": -2 }
}
```

Materials are `open`, `matte` (albedo), `mirror` (reflect, transmit) or `screen` (reflect,
transmit, emit). Reflect, transmit and emit take a number or an `[r, g, b]` triple.

Face and edge indices come from the hull, which is deterministic for a given vertex set;
the face dropdown in the panel lists them.

## Status

Renderer, polyhedral shells, cutting-plane mirrors, edge and free LED strips, click-to-select
editor, addressable LED patterns with a GLSL editor and LED map export, animated materials
and display panels, stateful fire, fog, lasers, STL shells. See `BRIEF.md` for scope and what comes next.
