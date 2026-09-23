// CPU picking against the compiled scene: hull faces (open ones too, so they can be made
// into mirrors), cut outlines, and every shell edge and strip as a thin capsule with a
// pick radius that grows with distance so it stays a few pixels wide on screen.

import { Compiled, Sel, V3, add, cross, dot, mul, sub } from './scene';
import { CameraBasis } from './tracer';

function planeRing(ro: V3, rd: V3, ring: V3[], n: V3): number {
  const dn = dot(rd, n);
  if (Math.abs(dn) < 1e-9) return -1;
  const t = (dot(n, ring[0]) - dot(n, ro)) / dn;
  if (t <= 0) return -1;
  const p = add(ro, mul(rd, t));
  for (let i = 0; i < ring.length; i++) {
    const a = ring[i], b = ring[(i + 1) % ring.length];
    if (dot(cross(sub(b, a), sub(p, a)), n) < 0) return -1;
  }
  return t;
}

/** ray vs capsule; returns t or -1 */
export function capsule(ro: V3, rd: V3, pa: V3, pb: V3, r: number): number {
  const ba = sub(pb, pa), oa = sub(ro, pa);
  const baba = dot(ba, ba), bard = dot(ba, rd), baoa = dot(ba, oa), rdoa = dot(rd, oa), oaoa = dot(oa, oa);
  const a = baba - bard * bard, b = baba * rdoa - baoa * bard, c = baba * oaoa - baoa * baoa - r * r * baba;
  let h = b * b - a * c;
  if (h >= 0) {
    const t = (-b - Math.sqrt(h)) / Math.max(a, 1e-12);
    const y = baoa + t * bard;
    if (y > 0 && y < baba) return t;
    const oc = y <= 0 ? oa : sub(ro, pb);
    const b2 = dot(rd, oc), c2 = dot(oc, oc) - r * r;
    h = b2 * b2 - c2;
    if (h > 0) return -b2 - Math.sqrt(h);
  }
  return -1;
}

/** primary ray through a pixel (client coords) */
export function mouseRay(cam: CameraBasis, x: number, y: number, w: number, h: number): { ro: V3; rd: V3 } {
  const nx = (x / w) * 2 - 1, ny = 1 - (y / h) * 2;
  const rd = add(add(cam.fwd, mul(cam.right, nx * cam.tanHalf * (w / h))), mul(cam.up, ny * cam.tanHalf));
  const L = Math.hypot(...rd);
  return { ro: cam.eye, rd: mul(rd, 1 / L) };
}

/**
 * pick the item under a ray. `pixel` is the world size of one pixel at unit distance;
 * thin items win over faces when they are within a few pixels of the ray.
 */
export function pick(c: Compiled, ro: V3, rd: V3, pixel: number): Sel | null {
  const PX = 6;
  let faceT = Infinity, faceSel: Sel | null = null;
  c.hull.faces.forEach((f, i) => {
    const t = planeRing(ro, rd, f.map(j => c.hull.verts[j]), c.hull.normals[i]);
    if (t > 0 && t < faceT) { faceT = t; faceSel = { kind: 'face', i }; }
  });
  c.cutRings.forEach((ring, i) => {
    if (!ring) return;
    const n = c.polys.find(p => p.ref.kind === 'cut' && p.ref.i === i)?.n
      ?? cross(sub(ring[1], ring[0]), sub(ring[2], ring[0]));
    const t = planeRing(ro, rd, ring, n);
    if (t > 0 && t < faceT) { faceT = t; faceSel = { kind: 'cut', i }; }
  });
  let thinT = Infinity, thinSel: Sel | null = null;
  const tryCap = (a: V3, b: V3, r: number, sel: Sel) => {
    // pick radius at the capsule's distance from the eye
    const dist = Math.max(1, dot(sub(mul(add(a, b), 0.5), ro), rd));
    const t = capsule(ro, rd, a, b, Math.max(r, pixel * dist * PX));
    if (t > 0 && t < thinT) { thinT = t; thinSel = sel; }
  };
  c.hull.edges.forEach((e, i) => tryCap(c.hull.verts[e.a], c.hull.verts[e.b], 0, { kind: 'edge', i }));
  c.caps.forEach(cp => tryCap(cp.a, cp.b, cp.radius, cp.ref));
  if (thinSel && thinT <= faceT + pixel * thinT * PX * 2) return thinSel;
  return faceSel;
}
