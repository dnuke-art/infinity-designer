// Convex polyhedra. A shell is the convex hull of a vertex set: library solids and
// user-supplied vertex lists go through the same code. The hull is brute force (every
// vertex triple that has all other vertices on one side is a face plane), which is
// plenty for the few dozen vertices these pieces have and has no degenerate cases to
// worry about beyond a tolerance.

import { V3, add, cross, dot, len, mul, norm, sub } from './scene';

export interface Hull {
  verts: V3[];
  /** false for an imported mesh: faces are its triangles and may enclose a concave shape */
  convex: boolean;
  /** convex hull of the vertices, for clipping cuts against a concave mesh (lazy) */
  outer?: Hull;
  /** vertex index rings, counter-clockwise seen from outside */
  faces: number[][];
  /** outward unit normals per face */
  normals: V3[];
  /** unordered vertex index pairs, plus the two faces each edge belongs to */
  edges: { a: number; b: number; faces: [number, number] }[];
}

const PHI = (1 + Math.sqrt(5)) / 2;

function signs(fn: (sx: number, sy: number, sz: number) => V3): V3[] {
  const out: V3[] = [];
  for (const sx of [-1, 1]) for (const sy of [-1, 1]) for (const sz of [-1, 1]) out.push(fn(sx, sy, sz));
  return out;
}
const dedupe = (vs: V3[]) => vs.filter((v, i) => vs.findIndex(w => len(sub(v, w)) < 1e-9) === i);

function ring(n: number, z: number, r = 1, phase = 0): V3[] {
  const out: V3[] = [];
  for (let i = 0; i < n; i++) {
    const a = phase + (i / n) * Math.PI * 2;
    out.push([r * Math.cos(a), r * Math.sin(a), z]);
  }
  return out;
}

/** library solids, unit-ish; oriented and normalised by `library()` */
const RAW: Record<string, () => V3[]> = {
  tetrahedron: () => [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]],
  cube: () => signs((x, y, z) => [x, y, z]),
  octahedron: () => [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
  dodecahedron: () => dedupe([
    ...signs((x, y, z) => [x, y, z]),
    ...signs((_, y, z) => [0, y / PHI, z * PHI]),
    ...signs((x, y) => [x / PHI, y * PHI, 0]),
    ...signs((x, _, z) => [x * PHI, 0, z / PHI]),
  ]),
  icosahedron: () => dedupe([
    ...signs((_, y, z) => [0, y, z * PHI]),
    ...signs((x, y) => [x, y * PHI, 0]),
    ...signs((x, _, z) => [x * PHI, 0, z]),
  ]),
};
for (let n = 3; n <= 8; n++) {
  RAW[`prism${n}`] = () => [...ring(n, -1, 1, Math.PI / 2), ...ring(n, 1, 1, Math.PI / 2)];
  RAW[`pyramid${n}`] = () => [...ring(n, -1, 1, Math.PI / 2), [0, 0, 1]];
  RAW[`antiprism${n}`] = () => [...ring(n, -0.5, 1, Math.PI / 2), ...ring(n, 0.5, 1, Math.PI / 2 + Math.PI / n)];
}
export const LIBRARY = Object.keys(RAW);

/** rotation taking unit vector a onto unit vector b, applied to p */
function rotateTo(p: V3, a: V3, b: V3): V3 {
  const ax = cross(a, b), s = len(ax), c = dot(a, b);
  if (s < 1e-9) return c > 0 ? p : [-p[0], -p[1], p[2]];
  const k = mul(ax, 1 / s), ang = Math.atan2(s, c);
  return add(add(mul(p, Math.cos(ang)), mul(cross(k, p), Math.sin(ang))), mul(k, dot(k, p) * (1 - Math.cos(ang))));
}

/** unit vertices for a library solid, a face turned to +z, scaled so max |coord| = 1 */
export function library(name: string): V3[] {
  const gen = RAW[name];
  if (!gen) throw new Error(`unknown polyhedron "${name}"; try ${LIBRARY.join(', ')}`);
  let vs = gen();
  const h = hull(vs);
  // turn the face whose normal is closest to +z exactly onto +z (prisms already are)
  let best = 0;
  h.normals.forEach((n, i) => { if (n[2] > h.normals[best][2]) best = i; });
  const n = h.normals[best];
  if (n[2] < 0.9999) vs = vs.map(p => rotateTo(p, n, [0, 0, 1]));
  // then turn about z so that face's first edge runs along +x, so the piece sits level
  if (!/^(prism|pyramid|antiprism)/.test(name)) {
    const f = h.faces[best];
    const e = sub(vs[f[1]], vs[f[0]]);
    const ang = -Math.atan2(e[1], e[0]);
    vs = vs.map(p => [p[0] * Math.cos(ang) - p[1] * Math.sin(ang), p[0] * Math.sin(ang) + p[1] * Math.cos(ang), p[2]]);
  }
  const m = Math.max(...vs.flat().map(Math.abs));
  return vs.map(p => mul(p, 1 / m));
}

/** order coplanar points counter-clockwise about their centroid seen from `n` */
export function orderRing(pts: V3[], n: V3): V3[] {
  const c = mul(pts.reduce((s, p) => add(s, p), [0, 0, 0] as V3), 1 / pts.length);
  const t = norm(sub(pts[0], c));
  const b = cross(n, t);
  return pts.map(p => ({ p, a: Math.atan2(dot(sub(p, c), b), dot(sub(p, c), t)) }))
    .sort((x, y) => x.a - y.a).map(x => x.p);
}

/** convex hull by brute force; tolerance relative to the model's extent */
export function hull(verts: V3[]): Hull {
  const n = verts.length;
  if (n < 4) throw new Error('a shell needs at least 4 vertices');
  const scale = Math.max(...verts.flat().map(Math.abs)) || 1;
  const eps = 1e-6 * scale;
  const planes: { n: V3; d: number; idx: number[] }[] = [];
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) for (let k = j + 1; k < n; k++) {
    let nn = cross(sub(verts[j], verts[i]), sub(verts[k], verts[i]));
    const L = len(nn);
    if (L < eps * eps) continue;
    nn = mul(nn, 1 / L);
    let d = dot(nn, verts[i]);
    let pos = 0, neg = 0;
    const on: number[] = [];
    for (let m = 0; m < n; m++) {
      const s = dot(nn, verts[m]) - d;
      if (s > eps) pos++; else if (s < -eps) neg++; else on.push(m);
    }
    if (pos && neg) continue;
    if (pos) { nn = mul(nn, -1); d = -d; }
    if (planes.some(p => dot(p.n, nn) > 1 - 1e-6 && Math.abs(p.d - d) < eps * 10)) continue;
    planes.push({ n: nn, d, idx: on });
  }
  if (planes.length < 4) throw new Error('vertices are coplanar or degenerate');
  const faces = planes.map(p => {
    const ordered = orderRing(p.idx.map(i => verts[i]), p.n);
    return ordered.map(q => p.idx.find(i => verts[i] === q)!);
  });
  const edgeMap = new Map<string, { a: number; b: number; faces: number[] }>();
  faces.forEach((f, fi) => f.forEach((a, i) => {
    const b = f[(i + 1) % f.length];
    const key = a < b ? `${a}-${b}` : `${b}-${a}`;
    const e = edgeMap.get(key) ?? { a: Math.min(a, b), b: Math.max(a, b), faces: [] };
    e.faces.push(fi);
    edgeMap.set(key, e);
  }));
  const edges = [...edgeMap.values()].filter(e => e.faces.length === 2)
    .map(e => ({ a: e.a, b: e.b, faces: [e.faces[0], e.faces[1]] as [number, number] }));
  return { verts, faces, normals: planes.map(p => p.n), edges, convex: true };
}

/** Newell normal of a ring, pointing away from `inside` */
export function faceNormal(ring: V3[], inside: V3): V3 {
  let n: V3 = [0, 0, 0];
  for (let i = 0; i < ring.length; i++) {
    const a = ring[i], b = ring[(i + 1) % ring.length];
    n = add(n, [(a[1] - b[1]) * (a[2] + b[2]), (a[2] - b[2]) * (a[0] + b[0]), (a[0] - b[0]) * (a[1] + b[1])]);
  }
  n = norm(n);
  const c = mul(ring.reduce((s, p) => add(s, p), [0, 0, 0] as V3), 1 / ring.length);
  return dot(n, sub(c, inside)) < 0 ? mul(n, -1) : n;
}

/** edges of an indexed face list: pairs shared by exactly two faces */
function edgesOf(faces: number[][]) {
  const edgeMap = new Map<string, { a: number; b: number; faces: number[] }>();
  faces.forEach((f, fi) => f.forEach((a, i) => {
    const b = f[(i + 1) % f.length];
    const key = a < b ? `${a}-${b}` : `${b}-${a}`;
    const e = edgeMap.get(key) ?? { a: Math.min(a, b), b: Math.max(a, b), faces: [] };
    e.faces.push(fi);
    edgeMap.set(key, e);
  }));
  return [...edgeMap.values()].filter(e => e.faces.length === 2)
    .map(e => ({ a: e.a, b: e.b, faces: [e.faces[0], e.faces[1]] as [number, number] }));
}

/** a Hull straight from an indexed mesh whose faces are already wound counter-clockwise outward */
export function meshHull(verts: V3[], faces: number[][]): Hull {
  const normals = faces.map(f => {
    const ring = f.map(i => verts[i]);
    let n: V3 = [0, 0, 0];
    for (let i = 0; i < ring.length; i++) {
      const a = ring[i], b = ring[(i + 1) % ring.length];
      n = add(n, [(a[1] - b[1]) * (a[2] + b[2]), (a[2] - b[2]) * (a[0] + b[0]), (a[0] - b[0]) * (a[1] + b[1])]);
    }
    return norm(n);
  });
  return { verts, faces, normals, edges: edgesOf(faces), convex: false };
}

/** parse a binary or ASCII STL into a mesh centred on its bounding box with max extent 1 */
export function parseSTL(buf: ArrayBuffer): { verts: V3[]; faces: number[][] } {
  const bytes = new Uint8Array(buf);
  const head = new TextDecoder().decode(bytes.slice(0, 512));
  const tris: { n: V3; v: [V3, V3, V3] }[] = [];
  const isAscii = head.startsWith('solid') && /facet/.test(head);
  if (isAscii) {
    const text = new TextDecoder().decode(bytes);
    const re = /facet\s+normal\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)[\s\S]*?vertex\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+vertex\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+vertex\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)/g;
    let m: RegExpExecArray | null;
    while ((m = re.exec(text))) {
      const f = m.slice(1).map(Number);
      tris.push({ n: [f[0], f[1], f[2]], v: [[f[3], f[4], f[5]], [f[6], f[7], f[8]], [f[9], f[10], f[11]]] });
    }
  } else {
    const dv = new DataView(buf);
    const n = dv.getUint32(80, true);
    for (let i = 0; i < n; i++) {
      const o = 84 + i * 50;
      const f = (k: number) => dv.getFloat32(o + k * 4, true);
      tris.push({ n: [f(0), f(1), f(2)], v: [[f(3), f(4), f(5)], [f(6), f(7), f(8)], [f(9), f(10), f(11)]] });
    }
  }
  if (!tris.length) throw new Error('no triangles in STL');
  const all = tris.flatMap(t => t.v);
  const mins = [0, 1, 2].map(k => Math.min(...all.map(v => v[k]))), maxs = [0, 1, 2].map(k => Math.max(...all.map(v => v[k])));
  const ctr = mins.map((lo, k) => (lo + maxs[k]) / 2), half = Math.max(...maxs.map((hi, k) => hi - mins[k])) / 2 || 1;
  const verts: V3[] = [], index = new Map<string, number>();
  const id = (v: V3) => {
    const q: V3 = [(v[0] - ctr[0]) / half, (v[1] - ctr[1]) / half, (v[2] - ctr[2]) / half];
    const key = q.map(x => x.toFixed(5)).join(',');
    let i = index.get(key);
    if (i === undefined) { i = verts.length; verts.push(q); index.set(key, i); }
    return i;
  };
  const faces: number[][] = [];
  for (const t of tris) {
    const ids = t.v.map(id);
    if (new Set(ids).size < 3) continue;
    const [a, b, c] = ids.map(i => verts[i]);
    let n = cross(sub(b, a), sub(c, a));
    if (dot(n, t.n) < 0 && len(t.n) > 0) ids.reverse();
    faces.push(ids);
  }
  return { verts, faces };
}

/** the polygon where the plane (unit normal n, dot(n, p) = d) cuts the hull; null if it misses */
export function clipPlane(h: Hull, n: V3, d: number): V3[] | null {
  if (!h.convex) h = h.outer ??= hull(h.verts);
  const scale = Math.max(...h.verts.flat().map(Math.abs)) || 1;
  const eps = 1e-7 * scale;
  const pts: V3[] = [];
  const s = h.verts.map(v => dot(n, v) - d);
  h.verts.forEach((v, i) => { if (Math.abs(s[i]) <= eps) pts.push(v); });
  for (const e of h.edges) {
    const sa = s[e.a], sb = s[e.b];
    if ((sa > eps && sb < -eps) || (sa < -eps && sb > eps)) {
      const t = sa / (sa - sb);
      pts.push(add(h.verts[e.a], mul(sub(h.verts[e.b], h.verts[e.a]), t)));
    }
  }
  const uniq = pts.filter((p, i) => pts.findIndex(q => len(sub(p, q)) < eps * 100) === i);
  if (uniq.length < 3) return null;
  return orderRing(uniq, n);
}

/** shrink a ring toward its centroid */
export function shrinkRing(ring: V3[], k: number): V3[] {
  const c = mul(ring.reduce((s, p) => add(s, p), [0, 0, 0] as V3), 1 / ring.length);
  return ring.map(p => add(c, mul(sub(p, c), k)));
}

/** ZYX Euler rotation in degrees */
export function euler(p: V3, deg: V3): V3 {
  const [ax, ay, az] = deg.map(a => a * Math.PI / 180);
  let [x, y, z] = p;
  let c = Math.cos(ax), s = Math.sin(ax);
  [y, z] = [y * c - z * s, y * s + z * c];
  c = Math.cos(ay); s = Math.sin(ay);
  [x, z] = [x * c + z * s, -x * s + z * c];
  c = Math.cos(az); s = Math.sin(az);
  [x, y] = [x * c - y * s, x * s + y * c];
  return [x, y, z];
}
