// Scene model. Units are millimetres, the shell is centred at the origin, +z faces the
// viewer, +y is up. A design is: a convex polyhedral shell whose faces are open, matte or
// mirror; cutting planes through the shell that become interior mirrors ("folding" the
// volume); LED strips along shell edges; and free LED strips anywhere. `compile()` turns
// that into the flat polygon + capsule lists the tracer and the picker consume.

import { Hull, clipPlane, euler, faceNormal, hull, library, meshHull, shrinkRing } from './poly';
import { HEART } from './heart';
import { Beam, Image, Laser, Sheet, traceLasers, virtualImages } from './laser';
export type { Laser } from './laser';

export type V3 = [number, number, number];

export interface Material {
  /** screen: an emissive panel (OLED, backlit LCD) with a glossy front; transmit > 0 makes it a transparent LCD */
  kind: 'open' | 'matte' | 'mirror' | 'screen';
  /** matte: diffuse albedo (black paint ~0.04) */
  albedo?: number;
  /** mirror/screen: fraction reflected (first-surface ~0.95, one-way film 0.3–0.7, display glass ~0.05); a triple tints per channel */
  reflect?: number | V3;
  /** mirror/screen: fraction transmitted; the rest is absorbed */
  transmit?: number | V3;
  /** screen: emitted radiance, same units as strip radiance */
  emit?: number | V3;
  /**
   * mirror/screen: GLSL `void material(Surf s, float t, inout vec3 R, inout vec3 T, inout vec3 E)`
   * that varies reflect / transmit / emission over the surface and in time (shutters,
   * switchable mirrors, pixel masks, display content). See mats.ts.
   */
  program?: string;
  programName?: string;
}

export interface Shell {
  /** library solid name, "custom" with `vertices` (convex hull), or "mesh" with `mesh` (any closed triangle mesh, e.g. an STL) */
  poly: string;
  vertices?: V3[];
  mesh?: { verts: V3[]; faces: number[][] };
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
  /** mm between LED centres (60/m = 16.67, 144/m = 6.94); 0 = continuous. LEDs per metre = 1000 / pitch */
  pitch: number;
  /** body / emitter radius, mm */
  radius: number;
  /** viewing angle in degrees (full angle at half intensity, the datasheet figure; bare SMD strips are ~120). 180 or more = omnidirectional */
  cone?: number;
  /**
   * radius in mm of a diffuser tube around the strip (neon-flex, opal channel); 0 = bare
   * LEDs. The tube glows with the LED dots blurred along it and emits in every direction.
   */
  tube?: number;
  /** diffusion length in mm along the tube (Gaussian sigma); default 1.5 × tube radius. Shorter shows hot spots */
  blur?: number;
  /** fraction of the light the diffuser lets out; opal ~0.7 */
  diffuserT?: number;
}

export interface EdgeStrip extends Partial<StripParams> {
  edge: number;
  /** how far inside the shell the strip sits, mm */
  inset?: number;
  /** data flows from the edge's higher-index vertex instead */
  reverse?: boolean;
  /**
   * width in mm of an opaque lip along this edge, lying in the plane of a face and reaching
   * inward across it, so the strip is hidden from that side and only its reflections show.
   * `shadeFace` picks the face; 'both' puts a lip on both faces (an opaque frame at the
   * edge, the strip shining inward from the corner). Left unset, the more transparent face
   * gets the lip, or both when they are equally see-through.
   */
  shade?: number;
  shadeFace?: number | 'both';
}

export interface Strip extends StripParams {
  name?: string;
  a: V3;
  b: V3;
  /** data flows from b to a */
  reverse?: boolean;
  /** emission axis when `cone` is set; default omnidirectional */
  normal?: V3;
}

export interface Fog {
  /** extinction per metre (scattering + absorption); 3 = a mean free path of 333 mm */
  density: number;
  /** fraction scattered rather than absorbed, per channel; 1 = white haze */
  albedo: V3;
  /** Henyey-Greenstein asymmetry, 0 isotropic, 0.5 forward-peaked like real haze */
  g: number;
}

export interface Pattern {
  name: string;
  /** GLSL body defining `vec3 pattern(Led led, float t)`, see leds.ts */
  source: string;
}

export interface Eye { pos: V3; target: V3; fov: number; /** display exposure, EV */ exposure?: number }

export interface Scene {
  name: string;
  shell: Shell;
  cuts: Cut[];
  edgeStrips: EdgeStrip[];
  /** defaults for edge strips that don't override them */
  edgeDefaults: StripParams & { inset: number; shade?: number };
  strips: Strip[];
  lasers?: Laser[];
  fog?: Fog;
  pattern?: Pattern;
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
  | { kind: 'cut'; i: number } | { kind: 'strip'; i: number } | { kind: 'laser'; i: number };

export interface Poly {
  ring: V3[]; n: V3; d: number;
  /** 0 matte, 1 mirror/screen, 2 portal (an open shell face, only marks the fog boundary), 3 laser sheet in fog */
  type: 0 | 1 | 2 | 3;
  /** shell face (crossing it enters or leaves the fog) */
  boundary: boolean;
  reflect: V3; transmit: V3; emit: V3; albedo: number;
  /** index into Compiled.programs when the material is animated, else -1 */
  mat: number;
  /** local frame for animated materials: centre, tangent·extent, bitangent·extent */
  centre: V3; tu: V3; tv: V3;
  ref: Sel;
}
export interface Cap {
  a: V3; b: V3;
  /** base colour (the "solid" pattern), for the overlay and the pattern's led.color */
  color: V3;
  radiance: number; pitch: number; radius: number;
  /** first global LED index and LED count, in wiring order from a to b; base -1 for non-LED emitters */
  base: number; count: number;
  /** 0 LED strip (colours from the pattern), 1 static emitter (laser line on a surface), 2 laser beam glowing in fog */
  kind: 0 | 1 | 2;
  /** emission axis (unit) and half-intensity half angle in radians; cosHalf -1 = omnidirectional */
  dir: V3; cosHalf: number;
  /** diffuser: blur sigma along the strip in mm (0 = bare LEDs) and the diffuser's transmission */
  blur: number; diffuserT: number;
  ref: Sel;
}
export interface Compiled {
  hull: Hull;
  polys: Poly[];
  caps: Cap[];
  /** total addressable LEDs */
  ledCount: number;
  /** animated material programs, in Poly.mat order */
  programs: { source: string; ref: Sel }[];
  beams: Beam[];
  sheets: Sheet[];
  /** virtual images of the emitters for lighting the fog */
  images: Image[];
  /** clipped outline per cut (null when the plane misses the shell) */
  cutRings: (V3[] | null)[];
}

export function shellHull(shell: Shell): Hull {
  const st = shell.stretch ?? [1, 1, 1], rot = shell.rotate ?? [0, 0, 0];
  const xf = (p: V3) => euler(mul([p[0] * st[0], p[1] * st[1], p[2] * st[2]], shell.size), rot);
  if (shell.poly === 'mesh') {
    const m = shell.mesh ?? HEART;
    return meshHull(m.verts.map(xf), m.faces.map(f => (st[0] * st[1] * st[2] < 0 ? [...f].reverse() : f)));
  }
  const unit = shell.poly === 'custom' ? (shell.vertices ?? []) : library(shell.poly);
  const verts = unit.map(xf);
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

const v3of = (x: number | V3 | undefined, dflt: number): V3 =>
  x === undefined ? [dflt, dflt, dflt] : typeof x === 'number' ? [x, x, x] : x;

/** the polygon's local frame: tangent follows world x projected onto the plane (y if the plane is x-facing) */
function frame(ring: V3[], n: V3) {
  const centre = mul(ring.reduce((s, p) => add(s, p), [0, 0, 0] as V3), 1 / ring.length);
  const axis: V3 = Math.abs(n[0]) > 0.9 ? [0, 1, 0] : [1, 0, 0];
  const t = norm(sub(axis, mul(n, dot(axis, n))));
  const b = cross(n, t);
  let eu = 0, ev = 0;
  for (const p of ring) { eu = Math.max(eu, Math.abs(dot(sub(p, centre), t))); ev = Math.max(ev, Math.abs(dot(sub(p, centre), b))); }
  return { centre, tu: mul(t, eu || 1), tv: mul(b, ev || 1) };
}

export function compile(scene: Scene): Compiled {
  const h = shellHull(scene.shell);
  const polys: Poly[] = [];
  const caps: Cap[] = [];
  const programs: { source: string; ref: Sel }[] = [];

  const pushPoly = (ring: V3[], n: V3, m: Material, ref: Sel, boundary: boolean) => {
    let mat = -1;
    const spec = m.kind === 'mirror' || m.kind === 'screen';
    if (spec && m.program) { mat = programs.length; programs.push({ source: m.program, ref }); }
    polys.push({
      ring, n, d: dot(n, ring[0]),
      type: m.kind === 'open' ? 2 : spec ? 1 : 0, boundary,
      reflect: spec ? v3of(m.reflect, m.kind === 'screen' ? 0.05 : 0.95) : [0, 0, 0],
      transmit: spec ? v3of(m.transmit, 0) : [0, 0, 0],
      emit: m.kind === 'screen' ? v3of(m.emit, 10) : [0, 0, 0],
      albedo: m.kind === 'matte' ? (m.albedo ?? 0.04) : 0,
      mat, ...frame(ring, n), ref,
    });
  };

  h.faces.forEach((f, i) => {
    const m = scene.shell.faces[i] ?? scene.shell.defaultFace;
    // open faces stay as portals so the tracer knows where the fog ends
    pushPoly(f.map(j => h.verts[j]), h.normals[i], m, { kind: 'face', i }, true);
  });

  const cutRings: (V3[] | null)[] = scene.cuts.map((c, i) => {
    const n = norm(c.normal);
    let ring = clipPlane(h, n, c.offset);
    if (!ring) return null;
    if (c.scale !== undefined && c.scale !== 1) ring = shrinkRing(ring, c.scale);
    if (c.material.kind !== 'open') pushPoly(ring, n, c.material, { kind: 'cut', i }, false);
    return ring;
  });

  const D = scene.edgeDefaults;
  let base = 0;
  const coneOf = (deg: number | undefined) => deg === undefined || deg >= 180 ? -1 : Math.max(0.05, deg * Math.PI / 360);
  const pushCap = (a: V3, b: V3, p: StripParams, reverse: boolean | undefined, dir: V3, cosHalf: number, ref: Sel) => {
    if (reverse) [a, b] = [b, a];
    const count = p.pitch > 0 ? Math.floor(len(sub(b, a)) / p.pitch + 1e-6) + 1 : 1;
    const tube = p.tube ?? 0;
    caps.push({ a, b, color: p.color, radiance: p.radiance, pitch: p.pitch, radius: tube > 0 ? tube : p.radius, base, count, kind: 0,
      dir, cosHalf: tube > 0 ? -1 : cosHalf, blur: tube > 0 ? (p.blur ?? tube * 1.5) : 0, diffuserT: p.diffuserT ?? 0.7, ref });
    base += count;
  };
  /** how see-through a shell face is: open 1, else its mean transmit */
  const openness = (fi: number) => {
    const m = scene.shell.faces[fi] ?? scene.shell.defaultFace;
    if (m.kind === 'open') return 1;
    const t = m.transmit ?? 0;
    return typeof t === 'number' ? t : (t[0] + t[1] + t[2]) / 3;
  };
  scene.edgeStrips.forEach(es => {
    const e = h.edges[es.edge];
    if (!e) return;
    const inset = es.inset ?? D.inset;
    const inward = mul(norm(add(h.normals[e.faces[0]], h.normals[e.faces[1]])), -1);
    const a0 = h.verts[e.a], b0 = h.verts[e.b];
    const along = norm(sub(b0, a0));
    const a = add(add(a0, mul(inward, inset)), mul(along, inset));
    const b = add(add(b0, mul(inward, inset)), mul(along, -inset));
    const shade = es.shade ?? D.shade ?? 0;
    // which faces get a lip: the viewing face, or both when neither is clearly the viewing side.
    // With one lip the strip mounts on the other face and points along its inward normal;
    // with two it sits in the corner and points along the bisector.
    const o0 = openness(e.faces[0]), o1 = openness(e.faces[1]);
    const lipFaces: number[] = es.shadeFace === 'both' ? [...e.faces]
      : typeof es.shadeFace === 'number' && e.faces.includes(es.shadeFace) ? [es.shadeFace]
      : Math.abs(o0 - o1) < 1e-6 ? [...e.faces] : [o1 > o0 ? e.faces[1] : e.faces[0]];
    const cone = es.cone ?? D.cone;
    const dir = shade > 0 && lipFaces.length === 1 ? mul(h.normals[lipFaces[0] === e.faces[0] ? e.faces[1] : e.faces[0]], -1) : inward;
    pushCap(a, b, { color: es.color ?? D.color, radiance: es.radiance ?? D.radiance, pitch: es.pitch ?? D.pitch, radius: es.radius ?? D.radius,
      tube: es.tube ?? D.tube, blur: es.blur ?? D.blur, diffuserT: es.diffuserT ?? D.diffuserT },
      es.reverse, dir, coneOf(cone), { kind: 'edge', i: es.edge });
    for (const viewFace of shade > 0 ? lipFaces : []) {
      const nF = h.normals[viewFace];
      const centre = mul(h.faces[viewFace].reduce((acc, j) => add(acc, h.verts[j]), [0, 0, 0] as V3), 1 / h.faces[viewFace].length);
      let across = norm(cross(nF, along));
      if (dot(across, sub(centre, a0)) < 0) across = mul(across, -1);
      const lift = mul(nF, -0.5);
      const ring: V3[] = [add(a0, lift), add(b0, lift), add(add(b0, lift), mul(across, shade)), add(add(a0, lift), mul(across, shade))];
      const n = norm(cross(sub(ring[1], ring[0]), sub(ring[2], ring[0])));
      polys.push({ ring, n, d: dot(n, ring[0]), type: 0, boundary: false, reflect: [0, 0, 0], transmit: [0, 0, 0], emit: [0, 0, 0],
        albedo: 0.04, mat: -1, ...frame(ring, n), ref: { kind: 'edge', i: es.edge } });
    }
  });

  scene.strips.forEach((s, i) => pushCap(s.a, s.b, s, s.reverse,
    s.normal ? norm(s.normal) : [0, 0, 1], s.normal ? coneOf(s.cone) : -1, { kind: 'strip', i }));

  const sigmaT = scene.fog ? scene.fog.density / 1000 : 0;
  const { beams, sheets, spots } = traceLasers(scene.lasers ?? [], polys, caps, sigmaT);
  caps.push(...spots);
  for (const sh of sheets) {
    const n = norm(cross(sub(sh.ring[1], sh.ring[0]), sub(sh.ring[3], sh.ring[0])));
    polys.push({ ring: sh.ring, n, d: dot(n, sh.ring[0]), type: 3, boundary: false, reflect: [0, 0, 0], transmit: [0, 0, 0],
      emit: mul(sh.color, sh.power), albedo: 0, mat: -1, centre: sh.ring[0], tu: sh.dir, tv: [0, 0, 0], ref: { kind: 'laser', i: sh.laser } });
  }
  const images = scene.fog ? virtualImages(caps, polys) : [];

  return { hull: h, polys, caps, ledCount: base, programs, beams, sheets, images, cutRings };
}

/** world position of every LED in wiring order (what a controller's pixel map wants) */
export function ledMap(c: Compiled): V3[] {
  const out: V3[] = [];
  for (const cp of c.caps) {
    if (cp.kind !== 0) continue;
    const dir = norm(sub(cp.b, cp.a));
    if (cp.pitch <= 0) { out.push(mul(add(cp.a, cp.b), 0.5)); continue; }
    for (let k = 0; k < cp.count; k++) out.push(add(cp.a, mul(dir, k * cp.pitch)));
  }
  return out;
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

function shaded(): Scene {
  const s = classic();
  s.name = 'shaded infinity mirror (lips, 120° LEDs)';
  s.edgeDefaults = { ...s.edgeDefaults, cone: 120, shade: 22 };
  s.eye.exposure = 0;
  return s;
}

function tubes(): Scene {
  const s = classic();
  s.name = 'neon-flex tubes';
  s.edgeDefaults = { ...s.edgeDefaults, pitch: 1000 / 96, tube: 5, blur: 7, inset: 14, radiance: 22 };
  s.eye.exposure = -1.5;
  return s;
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

function switchable(): Scene {
  const s = classic();
  s.name = 'switchable front mirror (LC + reflective polarizer)';
  s.eye.exposure = -1;
  s.cuts[0] = { name: 'LC switchable mirror', normal: [0, 0, 1], offset: 28,
    material: { kind: 'mirror', reflect: 0.9, transmit: 0, programName: 'switchable mirror', program: MAT_SWITCHABLE } };
  return s;
}

function ringMask(): Scene {
  const s = classic();
  s.name = 'LCD ring mask in front';
  s.cuts.push({ name: 'transparent LCD mask', normal: [0, 0, 1], offset: 29.5,
    material: { kind: 'mirror', reflect: 0, transmit: 0.4, programName: 'ring mask', program: MAT_RING } });
  s.eye = { pos: [0, 0, 700], target: [0, 0, 0], fov: 32, exposure: -0.5 };
  return s;
}

/** hull face index of the classic box that is the back mirror */
const backFace = (s: Scene) => Object.keys(s.shell.faces).map(Number).find(k => s.shell.faces[k].kind === 'mirror')!;

function oledBack(): Scene {
  const s = classic();
  s.name = 'OLED back panel';
  s.shell.faces[backFace(s)] = { kind: 'screen', reflect: 0.05, transmit: 0, emit: 0, programName: 'plasma', program: MAT_PLASMA };
  s.eye = { pos: [60, 40, 700], target: [0, 0, 0], fov: 32, exposure: -1.5 };
  return s;
}

function infinityDisplay(): Scene {
  const s = classic();
  s.name = 'LCD behind a back half mirror';
  s.shell.faces[backFace(s)] = { kind: 'screen', reflect: 0.05, transmit: 0, emit: 0, programName: 'expanding rings', program: MAT_RINGS_E };
  s.cuts.push({ name: 'back half mirror', normal: [0, 0, 1], offset: -26, material: { kind: 'mirror', reflect: 0.5, transmit: 0.45 } });
  s.edgeStrips = [];
  s.eye = { pos: [60, 40, 700], target: [0, 0, 0], fov: 32, exposure: -1 };
  return s;
}

function lcdMask(): Scene {
  const s = classic();
  s.name = 'transparent LCD over the back mirror';
  s.cuts.push({ name: 'transparent LCD (colour filter)', normal: [0, 0, 1], offset: -24,
    material: { kind: 'mirror', reflect: 0, transmit: 0.4, programName: 'image mask', program: MAT_MASK_IMAGE } });
  s.eye = { pos: [0, 0, 700], target: [0, 0, 0], fov: 32, exposure: -0.5 };
  return s;
}

function halfBacklit(): Scene {
  const s = classic();
  s.name = 'transparent LCD, half backlit';
  s.cuts.push({ name: 'LCD, backlight removed, right half lit', normal: [0, 0, 1], offset: -22,
    material: { kind: 'screen', reflect: 0.05, transmit: 0.4, emit: 0, programName: 'half backlit', program: MAT_HALF_BACKLIT } });
  s.eye = { pos: [40, 30, 700], target: [0, 0, 0], fov: 32, exposure: -1 };
  return s;
}

export const PAT_FIRE = `// Fire2012 (Mark Kriegsman) per strip: heat cools, drifts up the strip, sparks at the base.
// Heat lives in the pattern's state channel from frame to frame. Flames rise in wiring order.
vec3 pattern(Led led, float t) {
  float i = led.index, k = led.k, n = led.n;
  float seed = i * 7.3 + t * 61.0;
  float heat = prevState(i) - hash(seed) * (2.16 / n + 0.008);          // cool (COOLING 55)
  float h1 = prevState(i - 1.0), h2 = prevState(i - 2.0);
  if (k >= 2.0) heat = (h1 + 2.0 * h2) / 3.0 - hash(seed + 1.0) * (2.16 / n + 0.008);   // drift up
  if (k < 3.0 && hash(seed + 3.0) < 0.16) heat += 0.63 + 0.37 * hash(seed + 9.0);        // spark (SPARKING 120)
  state = clamp(heat, 0.0, 1.0);
  return heatColor(state);
}`;
export const PAT_FIRE_NOISE = `// stateless fire: 1-D noise rising along each strip, hottest at the base
float vnoise(float x) { float i = floor(x), f = fract(x); return mix(hash(i), hash(i + 1.0), f * f * (3.0 - 2.0 * f)); }
vec3 pattern(Led led, float t) {
  float u = led.u;
  float n = vnoise(u * 6.0 - t * 3.0 + led.strip * 13.0) * 0.6 + vnoise(u * 13.0 - t * 5.0 + led.strip * 7.0) * 0.4;
  float heat = clamp((1.0 - u) * 1.3 * n + 0.1 * (1.0 - u), 0.0, 1.0);
  return heatColor(heat);
}`;

function fire(): Scene {
  const s = classic();
  s.name = 'fire mirror';
  s.pattern = { name: 'fire (Fire2012)', source: PAT_FIRE };
  // flames rise from the bottom corners: left edge counts upward, right edge is reversed
  const h = shellHull(s.shell);
  for (const es of s.edgeStrips) {
    const e = h.edges[es.edge];
    const a = h.verts[e.a], b = h.verts[e.b];
    if (Math.abs(a[1] - b[1]) > 1) es.reverse = b[1] < a[1] || undefined;     // vertical: run upward
  }
  s.edgeDefaults.radiance = 25;
  s.eye = { pos: [80, 40, 700], target: [0, 0, 0], fov: 32, exposure: -1.5 };
  return s;
}

function fogLaser(): Scene {
  const s = classic();
  s.name = 'fog + laser fan';
  s.shell.stretch = [1, 1, 0.5];          // 150 mm deep so the beams have room
  s.cuts[0].offset = 70;
  s.edgeDefaults.inset = 10;
  s.fog = { density: 4, albedo: [0.9, 0.9, 0.9], g: 0.5 };
  s.lasers = [
    { name: 'green line laser', pos: [-140, -130, 20], dir: [1, 0.35, -0.6], color: [0.2, 1, 0.3], power: 60, fan: 50, roll: 10, radius: 1, rays: 20 },
    { name: 'red beam', pos: [140, 120, 10], dir: [-1, -0.4, -0.5], color: [1, 0.15, 0.1], power: 10, fan: 0, radius: 1.2 },
  ];
  s.eye = { pos: [100, 60, 700], target: [0, 0, 0], fov: 34, exposure: -1 };
  return s;
}

function laserScope(): Scene {
  const s = kaleidoscope();
  s.name = 'laser kaleidoscope';
  s.strips = [];
  s.fog = { density: 2, albedo: [0.9, 0.9, 0.9], g: 0.5 };
  s.lasers = [
    { name: 'cyan line laser', pos: [0, -40, 60], dir: [0.3, 0.6, -1], color: [0.2, 0.9, 1], power: 60, fan: 60, roll: 20, radius: 0.8, rays: 24 },
    { name: 'magenta line laser', pos: [30, 30, 55], dir: [-0.5, -0.4, -1], color: [1, 0.2, 0.7], power: 50, fan: 40, roll: -50, radius: 0.8, rays: 20, discrete: true },
  ];
  s.eye = { pos: [30, 20, 380], target: [0, 0, 0], fov: 45, exposure: -1 };
  return s;
}

function heart(): Scene {
  const shell: Shell = { poly: 'mesh', mesh: HEART, size: 150, faces: {}, defaultFace: GLASS };
  return {
    name: 'mirror heart (decimated-78.stl)', shell, cuts: [],
    edgeStrips: allEdges(shell).map(edge => ({ edge })),
    edgeDefaults: { ...DEFAULTS, color: [1, 0.15, 0.25], radiance: 20, inset: 5 },
    strips: [],
    eye: { pos: [150, 80, 700], target: [0, 0, 0], fov: 32, exposure: -1 },
  };
}

function tintedDrift(): Scene {
  const s = classic();
  s.name = 'tinted film + hue drift';
  s.cuts[0].material = { kind: 'mirror', reflect: [0.42, 0.5, 0.58], transmit: [0.55, 0.45, 0.35] };
  s.shell.faces[Object.keys(s.shell.faces).map(Number).find(k => s.shell.faces[k].kind === 'mirror')!] =
    { kind: 'mirror', reflect: [0.85, 0.92, 0.97], transmit: 0 };
  s.pattern = { name: 'hue drift', source: `vec3 pattern(Led led, float t) {
  return hsv(fract(0.08 - t * 0.04), 0.85, 1.0);
}` };
  return s;
}

// material programs used by presets; the full list lives in mats.ts
export const MAT_SWITCHABLE = `void material(Surf s, float t, inout vec3 R, inout vec3 T, inout vec3 E) {
  // LC cell + reflective polarizer over polarized LEDs: lossless swap between mirror and window
  float k = 0.4 + 0.4 * sin(t * 1.5);
  R = vec3(0.9 * k);
  T = vec3(0.9 * (1.0 - k));
}`;
export const MAT_RING = `void material(Surf s, float t, inout vec3 R, inout vec3 T, inout vec3 E) {
  // transparent LCD in front of the glass: a dark ring shrinking toward the centre
  float r = length(s.uv);
  float ring = fract(r * 2.0 + t * 0.4);
  R = vec3(0.0);
  T = vec3(0.4 * smoothstep(0.12, 0.3, ring));
}`;
export const MAT_PLASMA = `void material(Surf s, float t, inout vec3 R, inout vec3 T, inout vec3 E) {
  // display content: a plasma. Radiance ~2 is a bright panel next to strips at 30
  vec2 p = s.uv * 3.0;
  float v = sin(p.x + t) + sin(p.y * 1.3 - t * 0.7) + sin(length(p) * 2.0 - t * 1.5);
  E = 2.0 * (0.5 + 0.5 * cos(v * 1.5 + vec3(0.0, 2.1, 4.2)));
}`;
export const MAT_RINGS_E = `void material(Surf s, float t, inout vec3 R, inout vec3 T, inout vec3 E) {
  // display content: rings expanding from the centre, a depth cue on a back panel
  float r = length(s.uv);
  float w = pow(0.5 + 0.5 * sin(r * 12.0 - t * 4.0), 8.0);
  E = 3.0 * w * mix(vec3(0.1, 0.3, 1.0), vec3(1.0, 0.4, 0.1), r);
}`;
export const MAT_HALF_BACKLIT = `void material(Surf s, float t, inout vec3 R, inout vec3 T, inout vec3 E) {
  // an LCD with its backlight removed, over the back mirror. The right half has a local
  // backlight behind it (a transparent light guide, so it stays partly see-through); the
  // left half is a plain colour filter over the mirror. One pixel image drives both.
  vec2 p = s.uv * 4.0;
  float v = sin(p.x * 1.2 + t) + sin(p.y - t * 0.8) + sin(length(p) * 1.5 - t * 1.2);
  vec3 pixel = 0.5 + 0.5 * cos(v * 1.4 + vec3(0.0, 2.1, 4.2));
  float lit = smoothstep(-0.02, 0.02, s.uv.x);   // backlit region
  E = 2.0 * pixel * lit;
  T = 0.4 * pixel * mix(1.0, 0.7, lit);          // light guide costs a little transmission
  R = vec3(0.05);
}`;
export const MAT_MASK_IMAGE = `void material(Surf s, float t, inout vec3 R, inout vec3 T, inout vec3 E) {
  // transparent LCD as a colour filter: the loaded image (uImage) scrolls across it
  vec2 q = s.uv * 0.5 + 0.5;
  q.x = fract(q.x + t * 0.05);
  T = 0.4 * texture(uImage, q).rgb;
  R = vec3(0.0);
}`;

export const PRESETS: (() => Scene)[] = [
  classic,
  shaded,
  tubes,
  tilted,
  switchable,
  ringMask,
  tintedDrift,
  oledBack,
  infinityDisplay,
  lcdMask,
  halfBacklit,
  fire,
  fogLaser,
  laserScope,
  heart,
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
    const mm = m as Material;
    const k = mm?.kind;
    if (k !== 'open' && k !== 'matte' && k !== 'mirror' && k !== 'screen') throw new Error(`${what}.kind must be open, matte, mirror or screen`);
    for (const f of ['reflect', 'transmit', 'emit'] as const) {
      const x = mm[f];
      if (x !== undefined && typeof x !== 'number' && !(Array.isArray(x) && x.length === 3)) throw new Error(`${what}.${f} must be a number or [r, g, b]`);
    }
    if (mm.program !== undefined && typeof mm.program !== 'string') throw new Error(`${what}.program must be a string`);
  };
  if (!o || typeof o !== 'object') throw new Error('scene must be an object');
  if (!o.shell) throw new Error('shell missing');
  num(o.shell.size, 'shell.size');
  if (o.shell.poly === 'custom') (o.shell.vertices ?? []).forEach((v, i) => v3(v, `shell.vertices[${i}]`));
  else if (o.shell.poly === 'mesh') {
    const m = o.shell.mesh;
    if (!m || !Array.isArray(m.verts) || !Array.isArray(m.faces)) throw new Error('shell.mesh needs verts and faces');
    m.verts.forEach((v, i) => v3(v, `shell.mesh.verts[${i}]`));
    m.faces.forEach((f, i) => { if (!Array.isArray(f) || f.length < 3 || f.some(k => !Number.isInteger(k) || k < 0 || k >= m.verts.length)) throw new Error(`shell.mesh.faces[${i}] is not a valid ring`); });
  }
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
  if (o.pattern !== undefined && typeof o.pattern.source !== 'string') throw new Error('pattern.source must be a string');
  (o.lasers ?? []).forEach((l, i) => {
    v3(l.pos, `lasers[${i}].pos`); v3(l.dir, `lasers[${i}].dir`); v3(l.color, `lasers[${i}].color`);
    num(l.power, `lasers[${i}].power`); num(l.fan, `lasers[${i}].fan`);
  });
  if (o.fog) { num(o.fog.density, 'fog.density'); v3(o.fog.albedo, 'fog.albedo'); num(o.fog.g, 'fog.g'); }
  if (!o.eye) throw new Error('eye missing');
  v3(o.eye.pos, 'eye.pos'); v3(o.eye.target, 'eye.target'); num(o.eye.fov, 'eye.fov');
  compile(o); // throws on degenerate geometry
  return o;
}
