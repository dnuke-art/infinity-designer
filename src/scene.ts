// Scene model. Units are millimetres, the shell is centred at the origin, +z faces the
// viewer, +y is up. A design is: a convex polyhedral shell whose faces are open, matte or
// mirror; cutting planes through the shell that become interior mirrors ("folding" the
// volume); LED strips along shell edges; and free LED strips anywhere. `compile()` turns
// that into the flat polygon + capsule lists the tracer and the picker consume.

import { Hull, clipPlane, euler, faceNormal, hull, library, shrinkRing } from './poly';

export type V3 = [number, number, number];

export interface Material {
  kind: 'open' | 'matte' | 'mirror';
  /** matte: diffuse albedo (black paint ~0.04) */
  albedo?: number;
  /** mirror: fraction reflected (first-surface ~0.95, one-way film 0.3–0.7) */
  reflect?: number;
  /** mirror: fraction transmitted; the rest is absorbed */
  transmit?: number;
}

export interface Shell {
  /** library solid name, or "custom" with `vertices` */
  poly: string;
  vertices?: V3[];
  /** half-width in mm: the unit solid spans ±1 on its largest axis */
  size: number;
  /** per-axis multiplier applied before `size` */
  stretch?: V3;
  /** Euler rotation, degrees, x then y then z */
  rotate?: V3;
  /** materials by hull face index; anything missing uses `defaultFace` */
  faces: Record<number, Material>;
  defaultFace: Material;
}

export interface Cut {
  name?: string;
  normal: V3;
  /** signed distance of the plane from the origin along `normal` */
  offset: number;
  /** shrink the clipped polygon toward its centre (1 = fills the shell) */
  scale?: number;
  material: Material;
}

export interface StripParams {
  /** linear RGB 0–1 */
  color: V3;
  radiance: number;
  /** mm between LED centres (60/m = 16.67); 0 = continuous */
  pitch: number;
  /** body / emitter radius, mm */
  radius: number;
}

export interface EdgeStrip extends Partial<StripParams> {
  edge: number;
  /** how far inside the shell the strip sits, mm */
  inset?: number;
}

export interface Strip extends StripParams {
  name?: string;
  a: V3;
  b: V3;
}

export interface Eye { pos: V3; target: V3; fov: number }

export interface Scene {
  name: string;
  shell: Shell;
  cuts: Cut[];
  edgeStrips: EdgeStrip[];
  /** defaults for edge strips that don't override them */
  edgeDefaults: StripParams & { inset: number };
  strips: Strip[];
  eye: Eye;
}

// --- vectors ---------------------------------------------------------------------------
export const add = (a: V3, b: V3): V3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const mul = (a: V3, s: number): V3 => [a[0] * s, a[1] * s, a[2] * s];
export const dot = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const cross = (a: V3, b: V3): V3 => [
  a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
export const len = (a: V3) => Math.hypot(a[0], a[1], a[2]);
export const norm = (a: V3): V3 => mul(a, 1 / (len(a) || 1));

// --- compiled form ---------------------------------------------------------------------
export type Sel =
  | { kind: 'face'; i: number } | { kind: 'edge'; i: number }
  | { kind: 'cut'; i: number } | { kind: 'strip'; i: number };

export interface Poly {
  ring: V3[]; n: V3; d: number;
  /** 0 matte, 1 mirror */
  type: 0 | 1;
  reflect: number; transmit: number; albedo: number;
  ref: Sel;
}
export interface Cap { a: V3; b: V3; color: V3; pitch: number; radius: number; ref: Sel }
export interface Compiled {
  hull: Hull;
  polys: Poly[];
  caps: Cap[];
  /** clipped outline per cut (null when the plane misses the shell) */
  cutRings: (V3[] | null)[];
}

export function shellHull(shell: Shell): Hull {
  const unit = shell.poly === 'custom' ? (shell.vertices ?? []) : library(shell.poly);
  const st = shell.stretch ?? [1, 1, 1], rot = shell.rotate ?? [0, 0, 0];
  const verts = unit.map(p => euler(mul([p[0] * st[0], p[1] * st[1], p[2] * st[2]], shell.size), rot));
  const h = hull(verts);
  // Newell normals after the transform, so a negative stretch or odd rotation can't flip us
  h.normals = h.faces.map(f => faceNormal(f.map(i => verts[i]), [0, 0, 0]));
  h.faces = h.faces.map((f, i) => {
    const ring = f.map(j => verts[j]);
    const e0 = sub(ring[1], ring[0]), e1 = sub(ring[2], ring[1]);
    return dot(cross(e0, e1), h.normals[i]) < 0 ? [...f].reverse() : f;
  });
  return h;
}

function material(m: Material) {
  return {
    type: (m.kind === 'mirror' ? 1 : 0) as 0 | 1,
    reflect: m.kind === 'mirror' ? (m.reflect ?? 0.95) : 0,
    transmit: m.kind === 'mirror' ? (m.transmit ?? 0) : 0,
    albedo: m.kind === 'matte' ? (m.albedo ?? 0.04) : 0,
  };
}

export function compile(scene: Scene): Compiled {
  const h = shellHull(scene.shell);
  const polys: Poly[] = [];
  const caps: Cap[] = [];

  h.faces.forEach((f, i) => {
    const m = scene.shell.faces[i] ?? scene.shell.defaultFace;
    if (m.kind === 'open') return;
    const ring = f.map(j => h.verts[j]);
    const n = h.normals[i];
    polys.push({ ring, n, d: dot(n, ring[0]), ...material(m), ref: { kind: 'face', i } });
  });

  const cutRings: (V3[] | null)[] = scene.cuts.map((c, i) => {
    const n = norm(c.normal);
    let ring = clipPlane(h, n, c.offset);
    if (!ring) return null;
    if (c.scale !== undefined && c.scale !== 1) ring = shrinkRing(ring, c.scale);
    if (c.material.kind !== 'open')
      polys.push({ ring, n, d: dot(n, ring[0]), ...material(c.material), ref: { kind: 'cut', i } });
    return ring;
  });

  const D = scene.edgeDefaults;
  scene.edgeStrips.forEach(es => {
    const e = h.edges[es.edge];
    if (!e) return;
    const inset = es.inset ?? D.inset;
    const inward = mul(norm(add(h.normals[e.faces[0]], h.normals[e.faces[1]])), -1);
    const a0 = h.verts[e.a], b0 = h.verts[e.b];
    const along = norm(sub(b0, a0));
    const a = add(add(a0, mul(inward, inset)), mul(along, inset));
    const b = add(add(b0, mul(inward, inset)), mul(along, -inset));
    caps.push({ a, b, color: mul(es.color ?? D.color, es.radiance ?? D.radiance),
      pitch: es.pitch ?? D.pitch, radius: es.radius ?? D.radius, ref: { kind: 'edge', i: es.edge } });
  });

  scene.strips.forEach((s, i) => caps.push({ a: s.a, b: s.b, color: mul(s.color, s.radiance),
    pitch: s.pitch, radius: s.radius, ref: { kind: 'strip', i } }));

  return { hull: h, polys, caps, cutRings };
}

// --- presets ---------------------------------------------------------------------------
const WARM: V3 = [1, 0.75, 0.45];
const MIRROR: Material = { kind: 'mirror', reflect: 0.95, transmit: 0 };
const GLASS: Material = { kind: 'mirror', reflect: 0.5, transmit: 0.45 };
const MATTE: Material = { kind: 'matte', albedo: 0.04 };
const OPEN: Material = { kind: 'open' };
const DEFAULTS = { color: WARM, radiance: 30, pitch: 16.67, radius: 1.5, inset: 10 };

/** indices of the hull faces whose normal is nearest a direction */
function faceToward(shell: Shell, dir: V3): number {
  const h = shellHull(shell);
  let best = 0;
  h.normals.forEach((n, i) => { if (dot(n, dir) > dot(h.normals[best], dir)) best = i; });
  return best;
}
/** indices of the edges belonging to a face */
function faceEdges(shell: Shell, face: number): number[] {
  const h = shellHull(shell);
  return h.edges.map((e, i) => e.faces.includes(face) ? i : -1).filter(i => i >= 0);
}
function allEdges(shell: Shell): number[] { return shellHull(shell).edges.map((_, i) => i); }

function classic(): Scene {
  const shell: Shell = { poly: 'cube', size: 150, stretch: [1, 1, 0.2], faces: {}, defaultFace: MATTE };
  const front = faceToward(shell, [0, 0, 1]), back = faceToward(shell, [0, 0, -1]);
  shell.faces = { [front]: OPEN, [back]: MIRROR };
  return {
    name: 'classic infinity mirror', shell,
    cuts: [{ name: 'front glass (50/50 film)', normal: [0, 0, 1], offset: 28, material: GLASS }],
    edgeStrips: faceEdges(shell, front).map(edge => ({ edge })),
    edgeDefaults: { ...DEFAULTS, inset: 12 },
    strips: [],
    eye: { pos: [120, 80, 700], target: [0, 0, 0], fov: 32 },
  };
}

function tilted(): Scene {
  const shell: Shell = { poly: 'cube', size: 150, stretch: [1, 1, 0.27], faces: {}, defaultFace: MATTE };
  const front = faceToward(shell, [0, 0, 1]);
  shell.faces = { [front]: OPEN };
  const cols: V3[] = [[1, 0.1, 0.1], [0.1, 1, 0.2], [0.2, 0.3, 1], [1, 1, 1]];
  const t = 4 * Math.PI / 180;
  return {
    name: 'tilted back mirror, RGB', shell,
    cuts: [
      { name: 'front glass', normal: [0, 0, 1], offset: 38, material: GLASS },
      { name: 'back mirror, 4° tilt', normal: [0, Math.sin(t), Math.cos(t)], offset: -30, material: MIRROR },
    ],
    edgeStrips: faceEdges(shell, front).map((edge, i) => ({ edge, color: cols[i % 4] })),
    edgeDefaults: { ...DEFAULTS, inset: 12 },
    strips: [],
    eye: { pos: [0, 0, 700], target: [0, 0, 0], fov: 32 },
  };
}

function solid(name: string, poly: string, size: number, eyeZ: number): Scene {
  const shell: Shell = { poly, size, faces: {}, defaultFace: GLASS };
  return {
    name, shell, cuts: [],
    edgeStrips: allEdges(shell).map(edge => ({ edge })),
    edgeDefaults: { ...DEFAULTS, radiance: 20, inset: 6 },
    strips: [],
    eye: { pos: [eyeZ * 0.35, eyeZ * 0.25, eyeZ], target: [0, 0, 0], fov: 30 },
  };
}

function kaleidoscope(): Scene {
  const shell: Shell = { poly: 'prism3', size: 120, stretch: [1, 1, 0.6], faces: {}, defaultFace: MIRROR };
  const front = faceToward(shell, [0, 0, 1]), back = faceToward(shell, [0, 0, -1]);
  shell.faces = { [front]: GLASS, [back]: MATTE };
  const cols: V3[] = [[1, 0.2, 0.6], [0.2, 0.9, 1], [1, 0.9, 0.3]];
  const z = -120 * 0.6 + 5;
  return {
    name: 'triangular kaleidoscope', shell, cuts: [], edgeStrips: [],
    edgeDefaults: { ...DEFAULTS, inset: 6 },
    strips: [0, 1, 2].map(i => {
      const a = (i * 120 + 90) * Math.PI / 180;
      return { name: `arm ${i}`, a: [0, 0, z], b: [80 * Math.cos(a), 80 * Math.sin(a), z],
        color: cols[i], radiance: 25, pitch: 16.67, radius: 1.5 };
    }),
    eye: { pos: [0, 0, 380], target: [0, 0, 0], fov: 45 },
  };
}

export const PRESETS: (() => Scene)[] = [
  classic,
  tilted,
  () => solid('infinity cube', 'cube', 100, 600),
  () => solid('infinity dodecahedron', 'dodecahedron', 120, 650),
  () => solid('infinity icosahedron', 'icosahedron', 120, 650),
  kaleidoscope,
];

// --- validation ------------------------------------------------------------------------
export function validate(s: unknown): Scene {
  const o = s as Scene;
  const v3 = (x: unknown, what: string): V3 => {
    if (!Array.isArray(x) || x.length !== 3 || !x.every(n => typeof n === 'number' && isFinite(n)))
      throw new Error(`${what}: expected [x, y, z]`);
    return x as V3;
  };
  const num = (x: unknown, what: string) => { if (typeof x !== 'number' || !isFinite(x)) throw new Error(`${what} must be a number`); };
  const mat = (m: unknown, what: string) => {
    const k = (m as Material)?.kind;
    if (k !== 'open' && k !== 'matte' && k !== 'mirror') throw new Error(`${what}.kind must be open, matte or mirror`);
  };
  if (!o || typeof o !== 'object') throw new Error('scene must be an object');
  if (!o.shell) throw new Error('shell missing');
  num(o.shell.size, 'shell.size');
  if (o.shell.poly === 'custom') (o.shell.vertices ?? []).forEach((v, i) => v3(v, `shell.vertices[${i}]`));
  else library(o.shell.poly);
  if (o.shell.stretch) v3(o.shell.stretch, 'shell.stretch');
  if (o.shell.rotate) v3(o.shell.rotate, 'shell.rotate');
  o.shell.faces ??= {};
  Object.entries(o.shell.faces).forEach(([k, m]) => mat(m, `shell.faces[${k}]`));
  mat(o.shell.defaultFace, 'shell.defaultFace');
  o.cuts ??= [];
  o.cuts.forEach((c, i) => { v3(c.normal, `cuts[${i}].normal`); num(c.offset, `cuts[${i}].offset`); mat(c.material, `cuts[${i}].material`); });
  o.edgeStrips ??= [];
  o.edgeStrips.forEach((e, i) => num(e.edge, `edgeStrips[${i}].edge`));
  o.edgeDefaults ??= { ...DEFAULTS };
  o.strips ??= [];
  o.strips.forEach((t, i) => {
    v3(t.a, `strips[${i}].a`); v3(t.b, `strips[${i}].b`); v3(t.color, `strips[${i}].color`);
    for (const k of ['radiance', 'pitch', 'radius'] as const) num(t[k], `strips[${i}].${k}`);
  });
  if (!o.eye) throw new Error('eye missing');
  v3(o.eye.pos, 'eye.pos'); v3(o.eye.target, 'eye.target'); num(o.eye.fov, 'eye.fov');
  compile(o); // throws on degenerate geometry
  return o;
}
