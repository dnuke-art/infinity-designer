// WebGL2 progressive path tracer. One fragment shader traces a single camera path per
// pixel per frame: it walks the ray through the scene, and at every mirror hit flips a
// coin to reflect, transmit, or die with probabilities (reflect, transmit, rest). Walls
// are diffuse with the same Russian-roulette trick (continue with probability albedo).
// Because each event is sampled with its own probability, the path carries a throughput of
// exactly 1 until it hits an emitter, so the estimate is unbiased and needs no weights.
// Frames accumulate into an RGBA32F ping-pong pair; alpha counts samples.
//
// The scene is packed into a 1-D RGBA32F texture: polygons, then strips, then a vertex pool.
//   polygon = 3 texels: (normal, d) (type, reflect, transmit, albedo) (vertBase, nVerts, 0, 0)
//   strip   = 3 texels: (a, pitch) (b, radius) (color·radiance, 0)
//   vertex  = 1 texel:  (x, y, z, 0)
// Polygons are convex rings, counter-clockwise about their normal.

import { Compiled, V3, cross, dot, norm, sub } from './scene';

const VS = `#version 300 es
void main() {
  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

const TRACE_FS = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
uniform sampler2D uScene, uPrev;
uniform int uNP, uNS, uVB, uFrame, uMaxBounce;
uniform vec2 uRes;
uniform vec3 uEye, uFwd, uRight, uUp;
uniform float uTanHalf, uAmbient;
out vec4 o;

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

vec3 cosineHemi(vec3 n) {
  float u1 = rnd(), u2 = rnd();
  float r = sqrt(u1), phi = 6.2831853 * u2;
  vec3 t = normalize(abs(n.x) > 0.5 ? cross(n, vec3(0, 1, 0)) : cross(n, vec3(1, 0, 0)));
  vec3 b = cross(n, t);
  return normalize(t * (r * cos(phi)) + b * (r * sin(phi)) + n * sqrt(max(0.0, 1.0 - u1)));
}

void main() {
  ivec2 px = ivec2(gl_FragCoord.xy);
  seed = uint(px.x) * 1973u + uint(px.y) * 9277u + uint(uFrame) * 26699u;
  pcg();

  // jittered primary ray
  vec2 jit = vec2(rnd(), rnd());
  vec2 ndc = ((vec2(px) + jit) / uRes) * 2.0 - 1.0;
  float aspect = uRes.x / uRes.y;
  vec3 rd = normalize(uFwd + uRight * (ndc.x * uTanHalf * aspect) + uUp * (ndc.y * uTanHalf));
  vec3 ro = uEye;

  vec3 rad = vec3(0.0);
  const float EPS = 0.05; // mm

  for (int bounce = 0; bounce < 256; bounce++) {
    if (bounce >= uMaxBounce) break;
    float tB = 1e30; int kind = -1; int idx = -1;

    for (int q = 0; q < uNP; q++) {
      int b0 = q * 3;
      vec4 pl = S(b0);
      float dn = dot(rd, pl.xyz);
      if (abs(dn) < 1e-7) continue;
      float t = (pl.w - dot(ro, pl.xyz)) / dn;
      if (t <= EPS || t >= tB) continue;
      vec3 p = ro + rd * t;
      vec4 vi = S(b0 + 2);
      int vb = uVB + int(vi.x), nv = int(vi.y);
      bool inside = true;
      for (int k = 0; k < nv; k++) {
        vec3 a = S(vb + k).xyz, b = S(vb + ((k + 1 == nv) ? 0 : k + 1)).xyz;
        if (dot(cross(b - a, p - a), pl.xyz) < 0.0) { inside = false; break; }
      }
      if (!inside) continue;
      tB = t; kind = 0; idx = q;
    }
    int sBase = uNP * 3;
    for (int s = 0; s < uNS; s++) {
      int b0 = sBase + s * 3;
      vec4 a = S(b0), b = S(b0 + 1);
      float t = capsule(ro, rd, a.xyz, b.xyz, b.w);
      if (t > EPS && t < tB) { tB = t; kind = 1; idx = s; }
    }
    if (kind < 0) break; // escaped into the dark room

    vec3 p = ro + rd * tB;
    if (kind == 1) {
      int b0 = sBase + idx * 3;
      vec4 a = S(b0), b = S(b0 + 1), col = S(b0 + 2);
      vec3 ba = b.xyz - a.xyz;
      float L = length(ba);
      float along = dot(p - a.xyz, ba) / max(L, 1e-6);
      float lit = 1.0;
      if (a.w > 0.0) {
        float k = floor(along / a.w + 0.5);
        float d = abs(along - k * a.w);
        lit = (d < b.w && k * a.w <= L) ? 1.0 : 0.0;
      }
      rad += col.xyz * lit;
      break; // the strip body is opaque
    }

    int b0 = idx * 3;
    vec4 pl = S(b0), mt = S(b0 + 1);
    vec3 nn = dot(rd, pl.xyz) < 0.0 ? pl.xyz : -pl.xyz; // normal facing the incoming ray
    if (mt.x > 0.5) {
      // mirror: reflect / transmit / absorb
      float r = rnd();
      if (r < mt.y) { rd = reflect(rd, nn); ro = p + nn * EPS; }
      else if (r < mt.y + mt.z) { ro = p - nn * EPS; }
      else break;
    } else {
      // matte wall: a little ambient so the frame reads, then a diffuse bounce w.p. albedo
      float albedo = mt.w;
      rad += vec3(albedo * uAmbient);
      if (rnd() < albedo) { rd = cosineHemi(nn); ro = p + nn * EPS; }
      else break;
    }
  }

  vec4 prev = uFrame == 0 ? vec4(0.0) : texelFetch(uPrev, px, 0);
  o = prev + vec4(rad, 1.0);
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

export class Tracer {
  readonly gl: WebGL2RenderingContext;
  private trace: WebGLProgram;
  private show: WebGLProgram;
  private sceneTex: WebGLTexture;
  private acc: [WebGLTexture, WebGLTexture];
  private fbo: [WebGLFramebuffer, WebGLFramebuffer];
  private w = 0; private h = 0;
  private nP = 0; private nS = 0; private vb = 0;
  private uni: Record<string, WebGLUniformLocation | null> = {};
  private vao: WebGLVertexArrayObject;
  frame = 0;
  maxBounce = 64;
  exposure = 1;
  ambient = 0.02;

  constructor(readonly canvas: HTMLCanvasElement) {
    const gl = canvas.getContext('webgl2', { antialias: false, preserveDrawingBuffer: false });
    if (!gl) throw new Error('WebGL2 not available');
    if (!gl.getExtension('EXT_color_buffer_float')) throw new Error('EXT_color_buffer_float not available');
    this.gl = gl;
    this.trace = program(gl, VS, TRACE_FS);
    this.show = program(gl, VS, SHOW_FS);
    for (const name of ['uScene', 'uPrev', 'uNP', 'uNS', 'uVB', 'uFrame', 'uMaxBounce', 'uRes', 'uEye', 'uFwd',
      'uRight', 'uUp', 'uTanHalf', 'uAmbient'])
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
      polys.push(...p.n, p.d, p.type, p.reflect, p.transmit, p.albedo, verts.length / 4, p.ring.length, 0, 0);
      for (const v of p.ring) verts.push(...v, 0);
    }
    this.nP = c.polys.length;
    const strips: number[] = [];
    for (const s of c.caps) strips.push(...s.a, s.pitch, ...s.b, s.radius, ...s.color, 0);
    this.nS = c.caps.length;
    this.vb = (polys.length + strips.length) / 4;
    const data = new Float32Array([...polys, ...strips, ...verts]);
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
    gl.uniform1i(this.uni.uVB, this.vb);
    gl.uniform1i(this.uni.uMaxBounce, this.maxBounce);
    gl.uniform2f(this.uni.uRes, this.w, this.h);
    gl.uniform3fv(this.uni.uEye, cam.eye);
    gl.uniform3fv(this.uni.uFwd, cam.fwd);
    gl.uniform3fv(this.uni.uRight, cam.right);
    gl.uniform3fv(this.uni.uUp, cam.up);
    gl.uniform1f(this.uni.uTanHalf, cam.tanHalf);
    gl.uniform1f(this.uni.uAmbient, this.ambient);
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
