# infinity-designer — kickoff brief

- **Problem:** Infinity mirror and LED pieces are designed blind. You can't see what a
  mirror arrangement does to the light until the glass is cut and the strips are glued in.
  Want a sandbox to place LEDs and mirrors in a volume, "fold" the space, and see the piece
  from a viewer's eye before building.
- **Done looks like:** A 3D box with planar mirrors (full and half-silvered, with real
  reflectance and transmission values) and LED strips placed inside. Rendered from a
  movable eye position with honest per-bounce dimming. A classic framed infinity mirror
  preset looks like a photo of the real thing. Add one angled mirror and watch the tunnel
  break into a kaleidoscope. A design is a JSON file you can send someone.
- **Not now:** Lasers. (Bank the idea: a laser through a line-generator lens, bounced
  through the mirror stack, draws bright folded polylines on every surface with no fog
  needed. That is a surface-drawing sim and fits on top of this renderer later.) Curved
  mirrors, dichroic glass, animated LED sequences, fabrication export, haze, an editor for
  anything you can't already say in the scene file.
- **First slice:** The renderer, no editing UI. Scene file (box, N planar mirrors with R/T,
  M LED strips, eye) drives a WebGL2 fragment-shader path tracer: per pixel, trace from
  the eye, bounce up to K times, choose reflect / transmit / absorb stochastically at each
  mirror, accumulate over frames. Validate against the classic infinity mirror.
  Mirror-placement editing is slice two, the way tiedyer did fold engine first, dye second.
- **Open question:** Whether a per-pixel bouncing tracer converges fast enough for
  interactive editing at the bounce depths an infinity mirror needs (20 to 40 visible
  images). If not, the fallback is method-of-images for parallel mirrors and tracing only
  for the angled ones.
- **Platform:** Browser, statically hostable (Vite + TypeScript, no backend), same as
  tiedyer. GPU-first: everything per-pixel lives in shaders so iteration on a design is
  instant and the design space can be explored, not just verified.

## Why ray tracing, not method of images

Virtual images tile cleanly only when the mirrors form a proper reflection group (a box,
a kaleidoscope prism). One tilted mirror in a box breaks the tiling. A small scene of
rectangles and emissive capsules is cheap to trace directly, and half-silvered surfaces
fall out of the same loop as a coin flip between reflect and transmit.
