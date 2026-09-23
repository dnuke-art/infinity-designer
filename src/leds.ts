// Addressable LEDs. Every LED has a global index in wiring order (strips in scene order,
// each running from its data-in end). A tiny fragment shader evaluates a user-written
// GLSL function once per LED into an RGBA32F texture, which the tracer samples when a ray
// hits an emitter. Editing the pattern never touches the geometry, and the pass costs a
// few hundred fragments, so it runs every frame.

import { Compiled, PAT_FIRE, PAT_FIRE_NOISE, ledMap, sub, len } from './scene';

export const LEDW = 1024;

const VS = `#version 300 es
void main() {
  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

/** what a pattern can use; shown in the editor as documentation */
export const PATTERN_API = `struct Led {
  float index;   // 0 … count-1, wiring order
  float count;   // total LEDs
  float strip;   // strip number, wiring order
  float u;       // 0 … 1 along this strip
  float k;       // index within this strip, from its data-in end
  float n;       // LEDs in this strip
  vec3  pos;     // mm, +z toward the viewer
  vec3  color;   // the strip's base colour
};
vec3  hsv(float h, float s, float v);   vec3 heatColor(float heat);
float hash(float n);   float hash(vec3 p);   const float PI;
float prevState(float index);   // this LED's state last frame (any LED by index)
vec3  prevColor(float index);   // its colour last frame
float state;                    // set it to carry a value to the next frame
vec3 pattern(Led led, float t)   // t in seconds; return linear RGB, 0–1`;

const HEAD = `#version 300 es
#define LEDW ${LEDW}
precision highp float;
uniform sampler2D uAttr, uPrev;
uniform int uH;
uniform float uTime, uCount;
out vec4 o;
struct Led { float index; float count; float strip; float u; float k; float n; vec3 pos; vec3 color; };
const float PI = 3.14159265;
float state = 0.0;
vec4 prevTexel(float index) {
  int i = int(clamp(index, 0.0, uCount - 1.0));
  return texelFetch(uPrev, ivec2(i % LEDW, i / LEDW), 0);
}
float prevState(float index) { return prevTexel(index).a; }
vec3 prevColor(float index) { return prevTexel(index).rgb; }
vec3 heatColor(float h) {
  h = clamp(h, 0.0, 1.0);
  return vec3(smoothstep(0.0, 0.3, h), smoothstep(0.3, 0.7, h) * 0.9, smoothstep(0.75, 1.0, h));
}
vec3 hsv(float h, float s, float v) {
  vec3 k = vec3(1.0, 2.0 / 3.0, 1.0 / 3.0);
  vec3 p = abs(fract(vec3(h) + k) * 6.0 - 3.0);
  return v * mix(vec3(1.0), clamp(p - 1.0, 0.0, 1.0), s);
}
float hash(float n) { return fract(sin(n * 12.9898) * 43758.5453); }
float hash(vec3 p) { return fract(sin(dot(p, vec3(12.9898, 78.233, 37.719))) * 43758.5453); }
#line 1
`;
const TAIL = `
void main() {
  ivec2 px = ivec2(gl_FragCoord.xy);
  vec4 a = texelFetch(uAttr, px, 0);
  vec4 b = texelFetch(uAttr, px + ivec2(0, uH), 0);
  vec4 c = texelFetch(uAttr, px + ivec2(0, 2 * uH), 0);
  Led led = Led(a.w, b.z, b.x, b.y, b.w, c.a, a.xyz, c.rgb);
  vec3 col = max(pattern(led, uTime), 0.0);
  o = vec4(col, state);
}`;

export const PATTERNS: { name: string; source: string }[] = [
  { name: 'solid', source: `vec3 pattern(Led led, float t) {
  return led.color;
}` },
  { name: 'rainbow along wiring', source: `vec3 pattern(Led led, float t) {
  return hsv(fract(led.index / led.count - t * 0.1), 1.0, 1.0);
}` },
  { name: 'rainbow by height', source: `vec3 pattern(Led led, float t) {
  return hsv(fract(led.pos.y / 250.0 - t * 0.15), 1.0, 1.0);
}` },
  { name: 'one colour per strip', source: `vec3 pattern(Led led, float t) {
  return hsv(fract(led.strip * 0.618 + t * 0.03), 0.9, 1.0);
}` },
  { name: 'chase', source: `vec3 pattern(Led led, float t) {
  float d = fract(led.index / led.count * 4.0 - t * 0.6);
  return led.color * pow(d, 6.0);
}` },
  { name: 'breathe per strip', source: `vec3 pattern(Led led, float t) {
  return led.color * (0.15 + 0.85 * pow(0.5 + 0.5 * sin(t * 2.0 + led.strip * 0.9), 2.0));
}` },
  { name: 'plane wave', source: `vec3 pattern(Led led, float t) {
  float w = 0.5 + 0.5 * sin(dot(led.pos, normalize(vec3(1.0, 0.6, 0.3))) * 0.05 - t * 3.0);
  return mix(vec3(0.02, 0.0, 0.1), vec3(0.2, 0.9, 1.0), pow(w, 3.0));
}` },
  { name: 'fire (Fire2012)', source: PAT_FIRE },
  { name: 'fire (noise)', source: PAT_FIRE_NOISE },
  { name: 'sparkle', source: `vec3 pattern(Led led, float t) {
  float s = step(0.96, hash(led.index + floor(t * 10.0) * 977.0));
  return mix(led.color * 0.08, vec3(1.0), s);
}` },
];

export class PatternPass {
  private prog: WebGLProgram | null = null;
  private uni: Record<string, WebGLUniformLocation | null> = {};
  private attr: WebGLTexture;
  private texes: [WebGLTexture, WebGLTexture];
  private fbos: [WebGLFramebuffer, WebGLFramebuffer];
  private cur = 0;
  /** the LED colours for this frame */
  get tex() { return this.texes[this.cur]; }
  private vao: WebGLVertexArrayObject;
  private h = 1;
  private count = 0;
  source = '';

  constructor(private gl: WebGL2RenderingContext) {
    this.attr = gl.createTexture()!;
    this.texes = [gl.createTexture()!, gl.createTexture()!];
    this.fbos = [gl.createFramebuffer()!, gl.createFramebuffer()!];
    this.vao = gl.createVertexArray()!;
    for (const t of [this.attr, ...this.texes]) {
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    }
    // 1×1 white LED textures until a scene arrives
    for (const t of this.texes) {
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, 1, 1, 0, gl.RGBA, gl.FLOAT, new Float32Array([1, 1, 1, 1]));
    }
    const err = this.setSource(PATTERNS[0].source);
    if (err) throw new Error(err);
  }

  /** compile a pattern; returns an error message or null and keeps the previous program on failure */
  setSource(src: string): string | null {
    const gl = this.gl;
    const mk = (type: number, code: string) => {
      const sh = gl.createShader(type)!;
      gl.shaderSource(sh, code); gl.compileShader(sh);
      if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
        const log = gl.getShaderInfoLog(sh) || 'shader error';
        gl.deleteShader(sh);
        throw new Error(log.replace(/\0/g, '').replace(/^ERROR: 0:/gm, 'line ').trim());
      }
      return sh;
    };
    try {
      const vs = mk(gl.VERTEX_SHADER, VS), fs = mk(gl.FRAGMENT_SHADER, HEAD + src + TAIL);
      const p = gl.createProgram()!;
      gl.attachShader(p, vs); gl.attachShader(p, fs); gl.linkProgram(p);
      if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p) || 'link error');
      if (this.prog) gl.deleteProgram(this.prog);
      this.prog = p;
      for (const n of ['uAttr', 'uPrev', 'uH', 'uTime', 'uCount']) this.uni[n] = gl.getUniformLocation(p, n);
      this.source = src;
      return null;
    } catch (e) { return (e as Error).message; }
  }

  /** upload per-LED attributes for a compiled scene */
  setScene(c: Compiled) {
    const gl = this.gl;
    const pts = ledMap(c);
    this.count = pts.length;
    this.h = Math.max(1, Math.ceil(this.count / LEDW));
    const W = LEDW, H = this.h;
    const data = new Float32Array(W * H * 3 * 4);
    let i = 0;
    c.caps.filter(cp => cp.kind === 0).forEach((cp, s) => {
      const L = len(sub(cp.b, cp.a)) || 1;
      for (let k = 0; k < cp.count; k++, i++) {
        const p = pts[i];
        const x = i % W, y = Math.floor(i / W);
        const u = cp.count > 1 ? (k * cp.pitch) / L : 0.5;
        data.set([p[0], p[1], p[2], i], (y * W + x) * 4);
        data.set([s, u, this.count, k], ((y + H) * W + x) * 4);
        data.set([...cp.color, cp.count], ((y + 2 * H) * W + x) * 4);
      }
    });
    gl.bindTexture(gl.TEXTURE_2D, this.attr);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, W, H * 3, 0, gl.RGBA, gl.FLOAT, data);
    for (let i = 0; i < 2; i++) {
      gl.bindTexture(gl.TEXTURE_2D, this.texes[i]);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, W, H, 0, gl.RGBA, gl.FLOAT, new Float32Array(W * H * 4));
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbos[i]);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.texes[i], 0);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  /** the current LED colours (rgb, state) for every LED index */
  read(): Float32Array {
    const gl = this.gl;
    const out = new Float32Array(LEDW * this.h * 4);
    if (!this.count) return out;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbos[this.cur]);
    gl.readPixels(0, 0, LEDW, this.h, gl.RGBA, gl.FLOAT, out);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return out;
  }

  /** evaluate the pattern at time t into the LED texture */
  run(t: number) {
    const gl = this.gl;
    if (!this.prog || !this.count) return;
    const prev = this.cur, next = prev ^ 1;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbos[next]);
    gl.viewport(0, 0, LEDW, this.h);
    gl.useProgram(this.prog);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.attr);
    gl.uniform1i(this.uni.uAttr, 0);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.texes[prev]);
    gl.uniform1i(this.uni.uPrev, 1);
    gl.uniform1i(this.uni.uH, this.h);
    gl.uniform1f(this.uni.uTime, t);
    gl.uniform1f(this.uni.uCount, this.count);
    this.cur = next;
    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.bindVertexArray(null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }
}
