// Lasers and virtual images, both computed on the CPU from the compiled polygons.
//
// A laser is a point and a direction; with a line-generator lens it is a fan of rays in a
// plane. Each ray is followed through the mirrors, splitting at half-silvered ones, until
// its power is negligible. Where a ray ends on a matte surface it leaves a spot, and
// adjacent fan rays that took the same route leave a line: those become thin emissive
// capsules the tracer already knows how to draw. In fog the beam itself glows, so each
// ray becomes a volumetric capsule and each pair of adjacent fan rays a volumetric sheet.
//
// Virtual images: for next-event lighting of the fog, every LED strip is reflected across
// sequences of mirrors so a scatter point can aim a shadow ray at an image and have the
// ray validate the sequence by actually bouncing off those mirrors.

import { Cap, Poly, V3, add, cross, dot, len, mul, norm, sub } from './scene';
import { capsule } from './pick';

export interface Laser {
  name?: string;
  pos: V3;
  dir: V3;
  /** linear RGB */
  color: V3;
  /** spot radiance on a white matte surface; not a physical unit */
  power: number;
  /** full fan angle in degrees; 0 = a plain beam */
  fan: number;
  /** rotation of the fan plane about the beam, degrees; 0 = the line lies flat (horizontal) */
  roll?: number;
  /** beam radius, mm */
  radius?: number;
  /** rays across the fan */
  rays?: number;
  /**
   * a diffraction grating or lenticular array instead of a Powell / cylindrical lens: the
   * fan is `rays` separate beams, spaced evenly in the sine of the angle like grating orders,
   * leaving dots on surfaces instead of a line
   */
  discrete?: boolean;
}

/** a beam segment in space */
export interface Beam { a: V3; b: V3; power: number; color: V3; laser: number }
/** a fan sheet between two adjacent rays that took the same route: ring a0 a1 b1 b0 */
export interface Sheet { ring: [V3, V3, V3, V3]; power: number; color: V3; dir: V3; laser: number }

interface RayHit { t: number; poly?: number; cap?: number }

function planeRing(ro: V3, rd: V3, ring: V3[], n: V3): number {
  const dn = dot(rd, n);
  if (Math.abs(dn) < 1e-9) return -1;
  const t = (dot(n, ring[0]) - dot(n, ro)) / dn;
  if (t <= 0.05) return -1;
  const p = add(ro, mul(rd, t));
  for (let i = 0; i < ring.length; i++) {
    const a = ring[i], b = ring[(i + 1) % ring.length];
    if (dot(cross(sub(b, a), sub(p, a)), n) < 0) return -1;
  }
  return t;
}

function nearest(ro: V3, rd: V3, polys: Poly[], caps: Cap[]): RayHit | null {
  let best: RayHit | null = null;
  polys.forEach((p, i) => {
    if (p.type === 3) return; // sheets are not surfaces
    const t = planeRing(ro, rd, p.ring, p.n);
    if (t > 0 && (!best || t < best.t)) best = { t, poly: i };
  });
  caps.forEach((c, i) => {
    if (c.kind === 2) return;
    const t = capsule(ro, rd, c.a, c.b, c.radius);
    if (t > 0.05 && (!best || t < best.t)) best = { t, cap: i };
  });
  return best;
}

const mean = (v: V3) => (v[0] + v[1] + v[2]) / 3;

interface Path { route: string; end: V3; endPoly: number; segs: { a: V3; b: V3; power: number }[] }

/** follow one ray; returns every path it splits into */
function follow(ro: V3, rd: V3, power: number, polys: Poly[], caps: Cap[], sigmaT: number, out: Path[], route = '', segs: { a: V3; b: V3; power: number }[] = [], depth = 0) {
  if (power < 0.004 || depth > 14) return;
  const h = nearest(ro, rd, polys, caps);
  if (!h) return;
  const p = add(ro, mul(rd, h.t));
  const seg = { a: ro, b: p, power };
  const powerAtEnd = power * Math.exp(-sigmaT * h.t);
  if (h.cap !== undefined) { out.push({ route: route + 'c', end: p, endPoly: -1, segs: [...segs, seg] }); return; }
  const poly = polys[h.poly!];
  if (poly.type === 0) { out.push({ route: route + `m${h.poly}`, end: p, endPoly: h.poly!, segs: [...segs, seg] }); return; }
  if (poly.type === 2) { out.push({ route: route + 'x', end: p, endPoly: -1, segs: [...segs, seg] }); return; } // left through an open face
  const n = dot(rd, poly.n) < 0 ? poly.n : mul(poly.n, -1);
  const R = mean(poly.reflect), T = mean(poly.transmit);
  if (R > 0.004) {
    const rr = sub(rd, mul(n, 2 * dot(rd, n)));
    follow(add(p, mul(n, 0.05)), rr, powerAtEnd * R, polys, caps, sigmaT, out, route + `r${h.poly}`, [...segs, seg], depth + 1);
  }
  if (T > 0.004) follow(add(p, mul(n, -0.05)), rd, powerAtEnd * T, polys, caps, sigmaT, out, route + `t${h.poly}`, [...segs, seg], depth + 1);
  if (R <= 0.004 && T <= 0.004) out.push({ route: route + `a${h.poly}`, end: p, endPoly: -1, segs: [...segs, seg] });
}

export function traceLasers(lasers: Laser[], polys: Poly[], caps: Cap[], sigmaT: number) {
  const beams: Beam[] = [], sheets: Sheet[] = [], spots: Cap[] = [];
  lasers.forEach((L, li) => {
    const dir = norm(L.dir);
    const w = L.radius ?? 1;
    const fan = L.fan * Math.PI / 180;
    const N = fan > 0 ? Math.max(2, Math.round(L.rays ?? 24)) : 1;
    // fan plane: "flat" is perpendicular to world up, then rolled about the beam
    let side = cross(dir, [0, 1, 0]);
    if (dot(side, side) < 1e-9) side = [1, 0, 0];
    side = norm(side);
    const up = cross(side, dir);
    const roll = (L.roll ?? 0) * Math.PI / 180;
    const axis = add(mul(side, Math.cos(roll)), mul(up, Math.sin(roll)));
    const rayPower = L.power / N;
    const paths: Path[][] = [];
    const discrete = !!L.discrete && N > 1;
    for (let j = 0; j < N; j++) {
      let ang = N === 1 ? 0 : -fan / 2 + (fan * j) / (N - 1);
      // grating orders sit at sin θ = m λ / d: even in sin θ, so the outer beams bunch up in angle
      if (discrete) ang = Math.asin(Math.sin(fan / 2) * (N === 1 ? 0 : (2 * j / (N - 1) - 1)));
      const rd = norm(add(mul(dir, Math.cos(ang)), mul(axis, Math.sin(ang))));
      const out: Path[] = [];
      follow(L.pos, rd, rayPower, polys, caps, sigmaT, out);
      paths.push(out);
      for (const pth of out) for (const s of pth.segs) beams.push({ a: s.a, b: s.b, power: s.power, color: L.color, laser: li });
      if (N === 1 || discrete) for (const pth of out) if (pth.endPoly >= 0)
        spots.push({ a: pth.end, b: pth.end, color: L.color, radiance: rayPower * N * polys[pth.endPoly].albedo, pitch: 0, radius: w * 1.5, base: -1, count: 1, kind: 1, dir: [0, 0, 1], cosHalf: -1, blur: 0, diffuserT: 1, ref: { kind: 'laser', i: li } });
    }
    // lines on surfaces and sheets in fog between adjacent rays with the same route
    for (let j = 0; j + 1 < N && !discrete; j++) {
      for (const p0 of paths[j]) {
        const p1 = paths[j + 1].find(x => x.route === p0.route);
        if (!p1) continue;
        if (p0.endPoly >= 0) {
          const alb = polys[p0.endPoly].albedo;
          // a fan ray's power lands along the line between neighbours: radiance ~ power / (line length · width)
          const L0 = len(sub(p1.end, p0.end)) || 1;
          spots.push({ a: p0.end, b: p1.end, color: L.color, radiance: (rayPower * alb * 40) / (L0 * 2 * w), pitch: 0, radius: w, base: -1, count: 1, kind: 1, dir: [0, 0, 1], cosHalf: -1, blur: 0, diffuserT: 1, ref: { kind: 'laser', i: li } });
        }
        for (let k = 0; k < Math.min(p0.segs.length, p1.segs.length); k++) {
          const s0 = p0.segs[k], s1 = p1.segs[k];
          sheets.push({ ring: [s0.a, s1.a, s1.b, s0.b], power: s0.power + s1.power, color: L.color, dir: norm(sub(s0.b, s0.a)), laser: li });
        }
      }
    }
    // plain beams and grating orders glow as volumetric capsules
    if (N === 1 || discrete) for (const b of beams.filter(x => x.laser === li))
      spots.push({ a: b.a, b: b.b, color: L.color, radiance: b.power, pitch: 0, radius: w, base: -1, count: 1, kind: 2, dir: [0, 0, 1], cosHalf: -1, blur: 0, diffuserT: 1, ref: { kind: 'laser', i: li } });
  });
  return { beams, sheets, spots };
}

export interface Image { a: V3; b: V3; cap: number; seq: number[] }

function reflectPt(p: V3, poly: Poly): V3 {
  const d = dot(p, poly.n) - poly.d;
  return sub(p, mul(poly.n, 2 * d));
}

/**
 * virtual images of every LED strip across mirror sequences, breadth first, within a
 * budget. seq is in shadow-ray order: the ray from the fog meets seq[0] first.
 */
export function virtualImages(caps: Cap[], polys: Poly[], budget = 1200, maxDepth = 6): Image[] {
  const mirrors = polys.map((p, i) => ({ p, i })).filter(x => x.p.type === 1 && mean(x.p.reflect) > 0.05);
  const emitters = caps.map((c, i) => ({ c, i })).filter(x => x.c.kind !== 2);
  const out: Image[] = [];
  let level: Image[] = emitters.map(e => ({ a: e.c.a, b: e.c.b, cap: e.i, seq: [] }));
  for (let d = 1; d <= maxDepth; d++) {
    const next: Image[] = [];
    for (const img of level) for (const m of mirrors) {
      if (img.seq.length && img.seq[0] === m.i) continue;
      next.push({ a: reflectPt(img.a, m.p), b: reflectPt(img.b, m.p), cap: img.cap, seq: [m.i, ...img.seq] });
    }
    if (!next.length || out.length + next.length > budget) break;
    out.push(...next);
    level = next;
  }
  return out;
}
