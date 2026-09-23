// Export the compiled scene at the current time for tools/blender_render.py: polygons with
// their reflect / transmit / emit, the shell as a fog container, every LED as a sphere or
// every diffuser tube as coloured segments, laser lines and beams, the camera and exposure.
// Millimetres throughout; the Blender script converts.

import { Compiled, Scene, V3, add, len, mul, norm, sub } from './scene';
import { CameraBasis } from './tracer';

export interface BlenderScene {
  name: string;
  exposure: number;
  camera: { pos: V3; target: V3; fov: number };
  fog: Scene['fog'] | null;
  shell: { verts: V3[]; faces: number[][] };
  polys: { ring: V3[]; kind: 'matte' | 'mirror' | 'portal'; albedo: number; reflect: V3; transmit: V3; emit: V3 }[];
  emitters: ({ type: 'led'; pos: V3; radius: number; color: V3 } | { type: 'tube' | 'body'; a: V3; b: V3; radius: number; color: V3 })[];
  beams: { a: V3; b: V3; radius: number; color: V3; radiance: number }[];
  notes: string[];
}

export function blenderExport(scene: Scene, c: Compiled, cam: CameraBasis, target: V3, fov: number, exposure: number,
  ledColour: (i: number) => V3, matCentre: (m: number) => { R: V3; T: V3; E: V3 }): BlenderScene {
  const polys: BlenderScene['polys'] = [];
  for (const p of c.polys) {
    if (p.type === 3) continue;
    let reflect = p.reflect, transmit = p.transmit, emit = p.emit;
    if (p.mat >= 0) { const m = matCentre(p.mat); reflect = m.R; transmit = m.T; emit = m.E; }
    polys.push({ ring: p.ring, kind: p.type === 2 ? 'portal' : p.type === 1 ? 'mirror' : 'matte', albedo: p.albedo, reflect, transmit, emit });
  }
  const emitters: BlenderScene['emitters'] = [];
  const beams: BlenderScene['beams'] = [];
  for (const cp of c.caps) {
    const dir = norm(sub(cp.b, cp.a)), L = len(sub(cp.b, cp.a));
    if (cp.kind === 2) { beams.push({ a: cp.a, b: cp.b, radius: cp.radius, color: cp.color, radiance: cp.radiance * (scene.fog ? scene.fog.density * 0.1 : 0.02) }); continue; }
    if (cp.kind === 1) { emitters.push({ type: 'tube', a: cp.a, b: cp.b, radius: cp.radius, color: mul(cp.color, cp.radiance) }); continue; }
    const colourAt = (k: number) => ledColour(cp.base + k);
    if (cp.blur > 0 && cp.pitch > 0) {
      // diffuser tube: segments coloured by the same Gaussian blur the tracer uses
      const seg = Math.max(2, cp.pitch / 2), n = Math.max(1, Math.ceil(L / seg));
      const gain = cp.radiance * cp.diffuserT * (cp.pitch / (cp.blur * 2.5066));
      for (let i = 0; i < n; i++) {
        const s0 = (i / n) * L, s1 = ((i + 1) / n) * L, sm = (s0 + s1) / 2;
        let acc: V3 = [0, 0, 0];
        const k0 = Math.floor(sm / cp.pitch), span = Math.ceil(3 * cp.blur / cp.pitch);
        for (let k = k0 - span; k <= k0 + span + 1; k++) {
          if (k < 0 || k >= cp.count) continue;
          const d = sm - k * cp.pitch;
          acc = add(acc, mul(colourAt(k), Math.exp(-0.5 * d * d / (cp.blur * cp.blur))));
        }
        emitters.push({ type: 'tube', a: add(cp.a, mul(dir, s0)), b: add(cp.a, mul(dir, s1)), radius: cp.radius, color: mul(acc, gain) });
      }
    } else if (cp.pitch > 0) {
      emitters.push({ type: 'body', a: cp.a, b: cp.b, radius: cp.radius * 0.8, color: [0, 0, 0] });
      for (let k = 0; k < cp.count; k++)
        emitters.push({ type: 'led', pos: add(cp.a, mul(dir, k * cp.pitch)), radius: cp.radius, color: mul(colourAt(k), cp.radiance) });
    } else emitters.push({ type: 'tube', a: cp.a, b: cp.b, radius: cp.radius, color: mul(colourAt(0), cp.radiance * (cp.blur > 0 ? cp.diffuserT : 1)) });
  }
  const notes = [
    'LEDs are omnidirectional spheres in Blender; the app\'s emission lobe and shade lips still occlude them physically',
    'laser fan sheets are not exported; beams and surface lines are emissive cylinders',
    'animated materials use their value at the panel centre at the export time',
  ];
  return {
    name: scene.name, exposure,
    camera: { pos: cam.eye, target, fov },
    fog: scene.fog ?? null,
    shell: { verts: c.hull.verts, faces: c.hull.faces },
    polys, emitters, beams, notes,
  };
}
