// WebGL2 progressive path tracer. One fragment shader traces a single camera path per
// pixel per frame: it walks the ray through the scene, and at every mirror hit flips a
// coin to reflect, transmit, or die with probabilities (reflect, transmit, rest). Walls
// are diffuse with the same Russian-roulette trick (continue with probability albedo).
// Because each event is sampled with its own probability, the path carries a throughput
// of 1 until a tinted mirror or the fog moves it, so the estimate is unbiased.
// Frames accumulate into an RGBA32F ping-pong pair; alpha counts samples.
//
// Fog: inside the shell the ray may scatter before reaching a surface (distance sampled
// from the extinction). At a scatter point the fog is lit by next-event estimation: a
// shadow ray aims at an LED, or at a virtual image of one, and validates the image by
// actually reflecting off the listed mirrors in order. Laser beams and fan sheets glow
// where a camera ray crosses them.
//
// The scene is packed into a 1-D RGBA32F texture: polygons, strips, NEE emitters, vertices.
//   polygon = 9 texels: (normal, d) (type, albedo, matIndex, boundary) (vertBase, nVerts, 0, 0)
//                       (reflect rgb, 0) (transmit rgb, 0) (tangent·extent, 0) (bitangent·extent, 0) (centre, 0) (emit rgb, 0)
//             type 0 matte, 1 mirror/screen, 2 portal (open shell face), 3 laser sheet (tangent = beam dir)
//   strip   = 5 texels: (a, pitch) (b, radius) (radiance, first LED index, LED count, kind) (colour rgb, thetaHalf) (emission axis, 0)
//             thetaHalf -1 = omnidirectional, else a datasheet-like lobe: Gaussian in angle, half
//             intensity at thetaHalf, fading out just behind the package plane
//             kind 0 LED strip (colours from the LED texture), 1 static emitter, 2 beam glowing in fog
//   emitter = 3 texels: (a', pitch) (b', 0) (strip index, sequence length, seq0 | seq1 << 10, seq2 | seq3 << 10)
//   vertex  = 1 texel:  (x, y, z, 0)
// A polygon with matIndex >= 0 takes reflect/transmit/emit from the animated-material atlas
// (mats.ts): three TILE×TILE tiles per material, reflect, transmit, emit left to right.

import { Compiled, V3, cross, dot, norm, sub } from './scene';
import { TILE } from './mats';

const VS = `#version 300 es
void main() {
  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

const TRACE_FS = `#version 300 es
#define TILE ${TILE}
precision highp float;
precision highp int;
precision highp sampler2D;
uniform sampler2D uScene, uPrev, uLed, uMat;
uniform int uNP, uNS, uNN, uVB, uNB, uFrame, uMaxBounce, uImgDepth;
uniform vec2 uRes;
uniform vec3 uEye, uFwd, uRight, uUp, uFogAlbedo;
uniform float uTanHalf, uAmbient, uKeep, uFogT, uFogG;
uniform bool uEyeInFog;
out vec4 o;

const int LEDW = 1024;
const float EPS = 0.05; // mm
const float PI = 3.14159265;

vec4 S(int i) { return texelFetch(uScene, ivec2(i, 0), 0); }

uint seed;
uint pcg() {
  seed = seed * 747796405u + 2891336453u;
  uint w = ((seed >> ((seed >> 28u) + 4u)) ^ seed) * 277803737u;
  return (w >> 22u) ^ w;
}
float rnd() { return float(pcg()) / 4294967296.0; }

// ray vs capsule (pa, pb, r): returns t or -1. After Inigo Quilez.
float capsule(vec3 ro, vec3 rd, vec3 pa, vec3 pb, float r) {
  vec3 ba = pb - pa, oa = ro - pa;
  float baba = dot(ba, ba), bard = dot(ba, rd), baoa = dot(ba, oa), rdoa = dot(rd, oa), oaoa = dot(oa, oa);
  float a = baba - bard * bard;
  float b = baba * rdoa - baoa * bard;
  float c = baba * oaoa - baoa * baoa - r * r * baba;
  float h = b * b - a * c;
  if (h >= 0.0) {
    float t = (-b - sqrt(h)) / max(a, 1e-9);
    float y = baoa + t * bard;
    if (y > 0.0 && y < baba) return t;
    vec3 oc = (y <= 0.0) ? oa : ro - pb;
    b = dot(rd, oc); c = dot(oc, oc) - r * r; h = b * b - c;
    if (h > 0.0) return -b - sqrt(h);
  }
  return -1.0;
}

vec3 frameT(vec3 n) { return normalize(abs(n.x) > 0.5 ? cross(n, vec3(0, 1, 0)) : cross(n, vec3(1, 0, 0))); }

vec3 cosineHemi(vec3 n) {
  float u1 = rnd(), u2 = rnd();
  float r = sqrt(u1), phi = 2.0 * PI * u2;
  vec3 t = frameT(n), b = cross(n, t);
  return normalize(t * (r * cos(phi)) + b * (r * sin(phi)) + n * sqrt(max(0.0, 1.0 - u1)));
}

// Henyey-Greenstein phase function and its sampler
float hg(float c) {
  float g = uFogG;
  return (1.0 - g * g) / (4.0 * PI * pow(max(1e-4, 1.0 + g * g - 2.0 * g * c), 1.5));
}
vec3 sampleHG(vec3 d) {
  float g = uFogG, u1 = rnd(), u2 = rnd(), c;
  if (abs(g) < 1e-3) c = 1.0 - 2.0 * u1;
  else { float sq = (1.0 - g * g) / (1.0 - g + 2.0 * g * u1); c = (1.0 + g * g - sq * sq) / (2.0 * g); }
  c = clamp(c, -1.0, 1.0);
  float s = sqrt(max(0.0, 1.0 - c * c)), phi = 2.0 * PI * u2;
  vec3 t = frameT(d), b = cross(d, t);
  return normalize(t * (s * cos(phi)) + b * (s * sin(phi)) + d * c);
}

// ray vs polygon q: t or -1
float hitPoly(int q, vec3 ro, vec3 rd, float tMax) {
  int b0 = q * 9;
  vec4 pl = S(b0);
  float dn = dot(rd, pl.xyz);
  if (abs(dn) < 1e-7) return -1.0;
  float t = (pl.w - dot(ro, pl.xyz)) / dn;
  if (t <= EPS || t >= tMax) return -1.0;
  vec3 p = ro + rd * t;
  vec4 vi = S(b0 + 2);
  int vb = uVB + int(vi.x), nv = int(vi.y);
  for (int k = 0; k < nv; k++) {
    vec3 a = S(vb + k).xyz, b = S(vb + ((k + 1 == nv) ? 0 : k + 1)).xyz;
    if (dot(cross(b - a, p - a), pl.xyz) < 0.0) return -1.0;
  }
  return t;
}

// reflect / transmit / emit of polygon record b0 at point p (static or from the atlas)
void surfaceRTE(int b0, vec3 p, out vec3 Rv, out vec3 Tv, out vec3 Ev) {
  vec4 mt = S(b0 + 1);
  Rv = S(b0 + 3).xyz; Tv = S(b0 + 4).xyz; Ev = S(b0 + 8).xyz;
  if (mt.z >= 0.0) {
    vec3 tu = S(b0 + 5).xyz, tv = S(b0 + 6).xyz, q = p - S(b0 + 7).xyz;
    vec2 uv = vec2(dot(q, tu) / dot(tu, tu), dot(q, tv) / dot(tv, tv)) * 0.5 + 0.5;
    ivec2 tx = ivec2(clamp(uv * float(TILE), 0.0, float(TILE) - 1.0));
    tx.y += int(mt.z) * TILE;
    Rv = texelFetch(uMat, tx, 0).xyz;
    Tv = texelFetch(uMat, tx + ivec2(TILE, 0), 0).xyz;
    Ev = texelFetch(uMat, tx + ivec2(2 * TILE, 0), 0).xyz;
  }
}

vec3 ledColour(int li) { return texelFetch(uLed, ivec2(li % LEDW, li / LEDW), 0).rgb; }

// emission lobe of a directional strip: cs = cos of the angle from its axis
float lobe(float cs, float thetaHalf) {
  float th = acos(clamp(cs, -1.0, 1.0));
  return exp(-0.693 * (th / thetaHalf) * (th / thetaHalf)) * smoothstep(-0.15, 0.05, cs);
}

// next-event estimate of the light arriving at fog point P (incoming ray direction rdIn)
vec3 nee(vec3 P, vec3 rdIn) {
  if (uNN == 0) return vec3(0.0);
  int e = min(int(rnd() * float(uNN)), uNN - 1);
  int nb = uNB + e * 3;
  vec4 A = S(nb), B = S(nb + 1), M = S(nb + 2);
  int ci = int(M.x), nSeq = int(M.y), seqA = int(M.z), seqB = int(M.w);
  int sBase = uNP * 9, cb = sBase + ci * 5;
  vec4 cm = S(cb + 2), cc = S(cb + 3);
  float radius = S(cb + 1).w;
  vec3 dirE = B.xyz - A.xyz;
  float Le = length(dirE);
  dirE /= max(Le, 1e-6);
  vec3 Q, L;
  float area;
  int count = int(cm.z);
  if (A.w > 0.0 && cm.w < 0.5) {
    int k = min(int(rnd() * float(count)), count - 1);
    Q = A.xyz + dirE * float(k) * A.w;
    L = ledColour(int(cm.y) + k) * cm.x;
    area = PI * radius * radius * float(count);
  } else {
    Q = A.xyz + dirE * rnd() * Le;
    L = cc.rgb * cm.x;
    area = 2.0 * radius * max(Le, 2.0 * radius);
  }
  vec3 toQ = Q - P;
  float D = length(toQ);
  if (D < 1e-3) return vec3(0.0);
  vec3 sd = toQ / D;
  float ph = hg(dot(rdIn, sd));
  // the emitter's cone, seen from P (light leaves Q along -sd)
  if (cc.w > -0.5) L *= lobe(dot(S(cb + 4).xyz, -sd), cc.w);
  // shadow walk: through half mirrors, off the expected mirrors in order, to the emitter
  vec3 w = vec3(1.0), so = P, sdir = sd;
  float rem = D;
  int step = 0;
  for (int it = 0; it < 24; it++) {
    float tB = rem - EPS; int kind = -1; int idx = -1;
    for (int q = 0; q < uNP; q++) {
      int t1 = int(S(q * 9 + 1).x);
      if (t1 >= 2) continue;
      float t = hitPoly(q, so, sdir, tB);
      if (t > 0.0) { tB = t; kind = 0; idx = q; }
    }
    for (int s = 0; s < uNS; s++) {
      int b0 = sBase + s * 5;
      if (S(b0 + 2).w > 1.5) continue;
      float t = capsule(so, sdir, S(b0).xyz, S(b0 + 1).xyz, S(b0 + 1).w);
      if (t > EPS && t < tB) { tB = t; kind = 1; idx = s; }
    }
    if (kind < 0) { if (step != nSeq) return vec3(0.0); break; }
    if (kind == 1) { if (idx == ci && step == nSeq) break; return vec3(0.0); }
    int b0 = idx * 9;
    vec4 pl = S(b0), mt = S(b0 + 1);
    if (mt.x < 0.5) return vec3(0.0);
    vec3 hp = so + sdir * tB;
    vec3 Rv, Tv, Ev;
    surfaceRTE(b0, hp, Rv, Tv, Ev);
    int expected = step >= nSeq ? -1 : step == 0 ? (seqA & 1023) : step == 1 ? (seqA >> 10) : step == 2 ? (seqB & 1023) : (seqB >> 10);
    vec3 nn = dot(sdir, pl.xyz) < 0.0 ? pl.xyz : -pl.xyz;
    if (idx == expected) { w *= Rv; sdir = reflect(sdir, nn); so = hp + nn * EPS; step++; }
    else { w *= Tv; so = hp - nn * EPS; if (max(Tv.x, max(Tv.y, Tv.z)) < 1e-3) return vec3(0.0); }
    rem -= tB;
  }
  return w * L * area / (D * D) * ph * exp(-uFogT * D) * float(uNN);
}

void main() {
  ivec2 px = ivec2(gl_FragCoord.xy);
  seed = uint(px.x) * 1973u + uint(px.y) * 9277u + uint(uFrame) * 26699u;
  pcg();

  vec2 jit = vec2(rnd(), rnd());
  vec2 ndc = ((vec2(px) + jit) / uRes) * 2.0 - 1.0;
  float aspect = uRes.x / uRes.y;
  vec3 rd = normalize(uFwd + uRight * (ndc.x * uTanHalf * aspect) + uUp * (ndc.y * uTanHalf));
  vec3 ro = uEye;

  vec3 rad = vec3(0.0);
  vec3 thr = vec3(1.0);
  bool inFog = uEyeInFog && uFogT > 0.0;
  bool scattered = false;
  int reflSince = 0;
  int sBase = uNP * 9;
  float sigmaS = uFogT * 1000.0; // per metre, for the beam glow

  for (int bounce = 0; bounce < 256; bounce++) {
    if (bounce >= uMaxBounce) break;
    float tB = 1e30; int kind = -1; int idx = -1;
    for (int q = 0; q < uNP; q++) {
      float t = hitPoly(q, ro, rd, tB);
      if (t > 0.0) { tB = t; kind = 0; idx = q; }
    }
    for (int s = 0; s < uNS; s++) {
      int b0 = sBase + s * 5;
      float t = capsule(ro, rd, S(b0).xyz, S(b0 + 1).xyz, S(b0 + 1).w);
      if (t > EPS && t < tB) { tB = t; kind = 1; idx = s; }
    }

    // fog: scatter before the surface?
    if (inFog && uFogT > 0.0) {
      float d = -log(1.0 - rnd()) / uFogT;
      if (d < tB) {
        vec3 sp = ro + rd * d;
        thr *= uFogAlbedo;
        rad += thr * nee(sp, rd);
        rd = sampleHG(rd);
        ro = sp;
        scattered = true;
        reflSince = 0;
        continue;
      }
    }
    if (kind < 0) break;

    vec3 p = ro + rd * tB;
    if (kind == 1) {
      int b0 = sBase + idx * 5;
      vec4 a = S(b0), b = S(b0 + 1), m = S(b0 + 2), cc = S(b0 + 3);
      vec3 ba = b.xyz - a.xyz;
      float L = length(ba);
      vec3 bd = ba / max(L, 1e-6);
      float along = dot(p - a.xyz, bd);
      if (m.w > 1.5) {
        // a beam glowing in the fog: pass through it
        if (inFog && !scattered) {
          float c = dot(bd, -rd);
          float sa = sqrt(max(1e-3, 1.0 - dot(bd, rd) * dot(bd, rd)));
          float pw = m.x * exp(-uFogT * clamp(along, 0.0, L));
          rad += thr * cc.rgb * pw * sigmaS * uFogAlbedo * hg(c) / (2.0 * b.w * sa) * 0.5;
        }
        ro = p + rd * (2.0 * b.w + EPS);
        continue;
      }
      if (!scattered || reflSince > uImgDepth) {
        vec3 c;
        if (m.w < 0.5) {
          float lit = 1.0;
          int k = 0;
          if (a.w > 0.0) {
            float kf = floor(along / a.w + 0.5);
            float dk = abs(along - kf * a.w);
            lit = (dk < b.w && kf * a.w <= L + 1e-3) ? 1.0 : 0.0;
            k = int(kf);
          }
          c = ledColour(int(m.y) + k) * lit;
        } else c = cc.rgb;
        // emission cone: Lambertian about the axis, nothing outside the cone
        if (cc.w > -0.5) c *= lobe(dot(S(b0 + 4).xyz, -rd), cc.w);
        rad += thr * c * m.x;
      }
      break; // emitter bodies are opaque
    }

    int b0 = idx * 9;
    vec4 pl = S(b0), mt = S(b0 + 1);
    int type = int(mt.x);
    vec3 nn = dot(rd, pl.xyz) < 0.0 ? pl.xyz : -pl.xyz;
    if (type == 3) {
      // laser fan sheet glowing in the fog
      if (inFog && !scattered) {
        vec4 vi = S(b0 + 2);
        int vb = uVB + int(vi.x);
        vec3 v0 = S(vb).xyz, v1 = S(vb + 1).xyz, v2 = S(vb + 2).xyz, v3 = S(vb + 3).xyz;
        vec3 bd = S(b0 + 5).xyz;
        float L = length(v3 - v0);
        float u = clamp(dot(p - v0, bd) / max(L, 1e-6), 0.0, 1.0);
        float w = max(0.5, mix(length(v1 - v0), length(v2 - v3), u));
        float pw = exp(-uFogT * u * L);
        float c = dot(bd, -rd);
        rad += thr * S(b0 + 8).xyz * pw * sigmaS * uFogAlbedo * hg(c) / (w * max(abs(dot(rd, pl.xyz)), 0.05)) * 0.5;
      }
      ro = p + rd * EPS;
      continue;
    }
    if (type == 2) { inFog = !inFog; ro = p + rd * EPS; continue; }
    if (type == 1) {
      vec3 Rv, Tv, Ev;
      surfaceRTE(b0, p, Rv, Tv, Ev);
      rad += thr * Ev;
      float pr = (Rv.x + Rv.y + Rv.z) / 3.0, pt = (Tv.x + Tv.y + Tv.z) / 3.0;
      float r = rnd();
      if (r < pr) { thr *= Rv / pr; rd = reflect(rd, nn); ro = p + nn * EPS; reflSince++; }
      else if (r < pr + pt) { thr *= Tv / pt; ro = p - nn * EPS; if (mt.w > 0.5) inFog = !inFog; }
      else break;
    } else {
      // matte wall: a little ambient so the frame reads, then a diffuse bounce w.p. albedo
      float albedo = mt.y;
      rad += thr * albedo * uAmbient;
      if (rnd() < albedo) { rd = cosineHemi(nn); ro = p + nn * EPS; }
      else break;
    }
  }

  vec4 prev = uFrame == 0 ? vec4(0.0) : texelFetch(uPrev, px, 0);
  o = prev * uKeep + vec4(rad, 1.0);
}`;

const SHOW_FS = `#version 300 es
precision highp float;
uniform sampler2D uAcc;
uniform float uExposure;
out vec4 o;
vec3 aces(vec3 x) {
  return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}
void main() {
  vec4 a = texelFetch(uAcc, ivec2(gl_FragCoord.xy), 0);
  vec3 c = a.rgb / max(a.a, 1.0) * uExposure;
  o = vec4(pow(aces(c), vec3(1.0 / 2.2)), 1.0);
}`;

function compile(gl: WebGL2RenderingContext, type: number, src: string) {
  const s = gl.createShader(type)!;
  gl.shaderSource(s, src); gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s) || 'shader error');
  return s;
}
function program(gl: WebGL2RenderingContext, vs: string, fs: string) {
  const p = gl.createProgram()!;
  gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, vs));
  gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p) || 'link error');
  return p;
}

export interface CameraBasis { eye: V3; fwd: V3; right: V3; up: V3; tanHalf: number }
export interface FogParams { sigmaT: number; albedo: V3; g: number; eyeInside: boolean }

export class Tracer {
  readonly gl: WebGL2RenderingContext;
  private trace: WebGLProgram;
  private show: WebGLProgram;
  private sceneTex: WebGLTexture;
  private acc: [WebGLTexture, WebGLTexture];
  private fbo: [WebGLFramebuffer, WebGLFramebuffer];
  private w = 0; private h = 0;
  private nP = 0; private nS = 0; private nN = 0; private vb = 0; private nb = 0; private imgDepth = 0;
  private uni: Record<string, WebGLUniformLocation | null> = {};
  private vao: WebGLVertexArrayObject;
  frame = 0;
  maxBounce = 64;
  exposure = 1;
  ambient = 0.02;
  /** accumulator retention per sample: 1 = converge, <1 = moving average for animation */
  keep = 1;
  /** LED colour texture, provided by the pattern pass */
  ledTex: WebGLTexture | null = null;
  /** animated-material atlas, provided by the material pass */
  matTex: WebGLTexture | null = null;
  fog: FogParams = { sigmaT: 0, albedo: [1, 1, 1], g: 0, eyeInside: false };

  constructor(readonly canvas: HTMLCanvasElement) {
    const gl = canvas.getContext('webgl2', { antialias: false, preserveDrawingBuffer: false });
    if (!gl) throw new Error('WebGL2 not available');
    if (!gl.getExtension('EXT_color_buffer_float')) throw new Error('EXT_color_buffer_float not available');
    this.gl = gl;
    this.trace = program(gl, VS, TRACE_FS);
    this.show = program(gl, VS, SHOW_FS);
    for (const name of ['uScene', 'uPrev', 'uLed', 'uMat', 'uNP', 'uNS', 'uNN', 'uVB', 'uNB', 'uFrame', 'uMaxBounce', 'uImgDepth',
      'uRes', 'uEye', 'uFwd', 'uRight', 'uUp', 'uTanHalf', 'uAmbient', 'uKeep', 'uFogT', 'uFogG', 'uFogAlbedo', 'uEyeInFog'])
      this.uni[name] = gl.getUniformLocation(this.trace, name);
    this.uni.uAcc = gl.getUniformLocation(this.show, 'uAcc');
    this.uni.uExposure = gl.getUniformLocation(this.show, 'uExposure');
    this.sceneTex = gl.createTexture()!;
    this.acc = [gl.createTexture()!, gl.createTexture()!];
    this.fbo = [gl.createFramebuffer()!, gl.createFramebuffer()!];
    this.vao = gl.createVertexArray()!;
  }

  /** (re)allocate accumulation buffers for the canvas' current size */
  resize(w: number, h: number) {
    if (w === this.w && h === this.h) return;
    const gl = this.gl;
    this.w = w; this.h = h;
    this.canvas.width = w; this.canvas.height = h;
    for (let i = 0; i < 2; i++) {
      gl.bindTexture(gl.TEXTURE_2D, this.acc[i]);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, w, h, 0, gl.RGBA, gl.FLOAT, null);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo[i]);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.acc[i], 0);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    this.reset();
  }

  reset() { this.frame = 0; }

  /** pack the compiled scene into the scene texture; resets accumulation */
  setScene(c: Compiled) {
    const polys: number[] = [], verts: number[] = [];
    for (const p of c.polys) {
      polys.push(...p.n, p.d, p.type, p.albedo, p.mat, p.boundary ? 1 : 0, verts.length / 4, p.ring.length, 0, 0,
        ...p.reflect, 0, ...p.transmit, 0, ...p.tu, 0, ...p.tv, 0, ...p.centre, 0, ...p.emit, 0);
      for (const v of p.ring) verts.push(...v, 0);
    }
    this.nP = c.polys.length;
    const strips: number[] = [];
    for (const s of c.caps) strips.push(...s.a, s.pitch, ...s.b, s.radius, s.radiance, s.base, s.count, s.kind, ...s.color, s.cosHalf, ...s.dir, 0);
    this.nS = c.caps.length;
    // NEE emitters: every real strip, then the virtual images
    const nee: number[] = [];
    c.caps.forEach((s, i) => { if (s.kind !== 2) nee.push(...s.a, s.pitch, ...s.b, 0, i, 0, 0, 0); });
    this.imgDepth = 0;
    for (const im of c.images) {
      const sq = im.seq;
      if (sq.length > 4) continue;
      this.imgDepth = Math.max(this.imgDepth, sq.length);
      nee.push(...im.a, c.caps[im.cap].pitch, ...im.b, 0, im.cap, sq.length,
        (sq[0] ?? 0) + ((sq[1] ?? 0) << 10), (sq[2] ?? 0) + ((sq[3] ?? 0) << 10));
    }
    this.nN = nee.length / 12;
    this.nb = (polys.length + strips.length) / 4;
    this.vb = this.nb + nee.length / 4;

    const data = new Float32Array([...polys, ...strips, ...nee, ...verts]);
    const texels = Math.max(1, data.length / 4);
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.sceneTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, texels, 1, 0, gl.RGBA, gl.FLOAT, data.length ? data : new Float32Array(4));
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    this.reset();
  }

  /** trace `spp` more samples per pixel (0 = none) and present the running average */
  render(cam: CameraBasis, spp = 1) {
    const gl = this.gl;
    if (!this.w) return;
    gl.bindVertexArray(this.vao);
    gl.viewport(0, 0, this.w, this.h);
    gl.useProgram(this.trace);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.sceneTex);
    gl.uniform1i(this.uni.uScene, 0);
    gl.uniform1i(this.uni.uNP, this.nP);
    gl.uniform1i(this.uni.uNS, this.nS);
    gl.uniform1i(this.uni.uNN, this.nN);
    gl.uniform1i(this.uni.uVB, this.vb);
    gl.uniform1i(this.uni.uNB, this.nb);
    gl.uniform1i(this.uni.uImgDepth, this.imgDepth);
    gl.uniform1i(this.uni.uMaxBounce, this.maxBounce);
    gl.uniform2f(this.uni.uRes, this.w, this.h);
    gl.uniform3fv(this.uni.uEye, cam.eye);
    gl.uniform3fv(this.uni.uFwd, cam.fwd);
    gl.uniform3fv(this.uni.uRight, cam.right);
    gl.uniform3fv(this.uni.uUp, cam.up);
    gl.uniform1f(this.uni.uTanHalf, cam.tanHalf);
    gl.uniform1f(this.uni.uAmbient, this.ambient);
    gl.uniform1f(this.uni.uKeep, this.keep);
    gl.uniform1f(this.uni.uFogT, this.fog.sigmaT);
    gl.uniform1f(this.uni.uFogG, this.fog.g);
    gl.uniform3fv(this.uni.uFogAlbedo, this.fog.albedo);
    gl.uniform1i(this.uni.uEyeInFog, this.fog.eyeInside ? 1 : 0);
    gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, this.ledTex);
    gl.uniform1i(this.uni.uLed, 2);
    gl.activeTexture(gl.TEXTURE3); gl.bindTexture(gl.TEXTURE_2D, this.matTex);
    gl.uniform1i(this.uni.uMat, 3);
    for (let i = 0; i < spp; i++) {
      const src = this.frame & 1, dst = src ^ 1;
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo[dst]);
      gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.acc[src]);
      gl.uniform1i(this.uni.uPrev, 1);
      gl.uniform1i(this.uni.uFrame, this.frame);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      this.frame++;
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.useProgram(this.show);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.acc[this.frame & 1]);
    gl.uniform1i(this.uni.uAcc, 0);
    gl.uniform1f(this.uni.uExposure, this.exposure);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }
}

/** camera basis from eye, target, vertical fov (deg) */
export function basis(eye: V3, target: V3, fovDeg: number): CameraBasis {
  const fwd = norm(sub(target, eye));
  let right = cross(fwd, [0, 1, 0]);
  if (dot(right, right) < 1e-9) right = [1, 0, 0];
  right = norm(right);
  const up = cross(right, fwd);
  return { eye, fwd, right, up, tanHalf: Math.tan(fovDeg * Math.PI / 360) };
}
