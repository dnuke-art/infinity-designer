// Animated materials. A mirror face or cut can carry a GLSL program giving per-channel
// reflect and transmit as a function of position on the surface and time: a shutter, a
// switchable mirror, a pixel mask, a tint that drifts, or an emissive display panel showing
// content. Every animated material gets three TILE×TILE tiles in an RGBA32F atlas (reflect,
// transmit, emit), re-evaluated each frame by one fragment shader that concatenates all
// the programs. `uImage` is a user-loaded picture any program can sample.

import { Compiled, MAT_HALF_BACKLIT, MAT_MASK_IMAGE, MAT_PLASMA, MAT_RINGS_E, MAT_RING, MAT_SWITCHABLE, Sel } from './scene';

export const TILE = 128;

const VS = `#version 300 es
void main() {
  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

export const MATERIAL_API = `struct Surf {
  vec2  uv;     // -1 … 1 across the polygon, u along world x, v along world y
  vec3  pos;    // mm
  float id;     // this material's index
};
float hash(float n);   const float PI;
uniform sampler2D uImage;   // the loaded picture, sample with s.uv * 0.5 + 0.5
void material(Surf s, float t, inout vec3 R, inout vec3 T, inout vec3 E)
  // R reflected, T transmitted (per channel, R + T ≤ 1), E emitted radiance.
  // They arrive holding the static values; change only what you need.`;

const HEAD = `#version 300 es
precision highp float;
uniform sampler2D uFrame, uImage;
uniform float uTime;
out vec4 o;
struct Surf { vec2 uv; vec3 pos; float id; };
const float PI = 3.14159265;
float hash(float n) { return fract(sin(n * 12.9898) * 43758.5453); }
`;

export const MATERIALS: { name: string; source: string }[] = [
  { name: 'switchable mirror', source: MAT_SWITCHABLE },
  { name: 'shutter (square wave)', source: `void material(Surf s, float t, inout vec3 R, inout vec3 T, inout vec3 E) {
  // LC shutter: open 0.4 (two polarizers), closed dark
  float open = step(0.5, fract(t * 0.5));
  R = vec3(0.0);
  T = vec3(0.4 * open);
}` },
  { name: 'breathing half mirror', source: `void material(Surf s, float t, inout vec3 R, inout vec3 T, inout vec3 E) {
  float k = 0.3 + 0.4 * (0.5 + 0.5 * sin(t));
  R = vec3(k);
  T = vec3(0.9 - k);
}` },
  { name: 'ring mask', source: MAT_RING },
  { name: 'wipe mask', source: `void material(Surf s, float t, inout vec3 R, inout vec3 T, inout vec3 E) {
  // a dark bar sweeping across a transparent LCD
  float x = fract(s.uv.x * 0.5 + 0.5 - t * 0.2);
  R = vec3(0.0);
  T = vec3(0.4 * smoothstep(0.1, 0.2, x));
}` },
  { name: 'tinted film', source: `void material(Surf s, float t, inout vec3 R, inout vec3 T, inout vec3 E) {
  R = vec3(0.42, 0.50, 0.58);
  T = vec3(0.55, 0.45, 0.35);
}` },
  { name: 'plasma (display)', source: MAT_PLASMA },
  { name: 'expanding rings (display)', source: MAT_RINGS_E },
  { name: 'image (display)', source: `void material(Surf s, float t, inout vec3 R, inout vec3 T, inout vec3 E) {
  E = 2.0 * texture(uImage, s.uv * 0.5 + 0.5).rgb;
}` },
  { name: 'image mask (transparent LCD)', source: MAT_MASK_IMAGE },
  { name: 'half backlit (transparent LCD)', source: MAT_HALF_BACKLIT },
  { name: 'colour-cycling film', source: `void material(Surf s, float t, inout vec3 R, inout vec3 T, inout vec3 E) {
  vec3 c = 0.5 + 0.5 * cos(t * 0.7 + vec3(0.0, 2.1, 4.2));
  R = mix(vec3(0.5), c, 0.35) * 0.9;
  T = 0.95 - R;
}` },
];

export class MaterialPass {
  private prog: WebGLProgram | null = null;
  private uni: Record<string, WebGLUniformLocation | null> = {};
  private frames: WebGLTexture;
  readonly tex: WebGLTexture;
  /** user picture available to programs as uImage */
  readonly image: WebGLTexture;
  private fbo: WebGLFramebuffer;
  private vao: WebGLVertexArrayObject;
  private count = 0;
  /** compile errors from the last setScene, keyed by program index */
  errors = new Map<number, string>();
  refs: Sel[] = [];

  constructor(private gl: WebGL2RenderingContext) {
    this.frames = gl.createTexture()!;
    this.tex = gl.createTexture()!;
    this.image = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, this.image);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([255, 255, 255, 255]));
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
    this.fbo = gl.createFramebuffer()!;
    this.vao = gl.createVertexArray()!;
    for (const t of [this.frames, this.tex]) {
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    }
    gl.bindTexture(gl.TEXTURE_2D, this.tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, 1, 1, 0, gl.RGBA, gl.FLOAT, new Float32Array([0, 0, 0, 1]));
  }

  /** upload frames and build one shader from every animated material's program */
  setScene(c: Compiled) {
    const gl = this.gl;
    this.errors.clear();
    this.refs = c.programs.map(p => p.ref);
    this.count = c.programs.length;
    if (!this.count) { if (this.prog) { gl.deleteProgram(this.prog); this.prog = null; } return; }

    const fr = new Float32Array(this.count * 6 * 4);
    c.polys.filter(p => p.mat >= 0).forEach(p =>
      fr.set([...p.centre, 0, ...p.tu, 0, ...p.tv, 0, ...p.reflect, 0, ...p.transmit, 0, ...p.emit, 0], p.mat * 24));
    gl.bindTexture(gl.TEXTURE_2D, this.frames);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, 6, this.count, 0, gl.RGBA, gl.FLOAT, fr);
    gl.bindTexture(gl.TEXTURE_2D, this.tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, TILE * 3, TILE * this.count, 0, gl.RGBA, gl.FLOAT, null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.tex, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    // each program is compiled alone first so an error names the right material, then all
    // are concatenated with `material` renamed to `material_<i>` and dispatched by tile row
    const bodies: string[] = [];
    const SIG = /void\s+material\s*\(\s*Surf\s+\w+\s*,\s*float\s+\w+\s*,\s*inout\s+vec3\s+\w+\s*,\s*inout\s+vec3\s+\w+\s*,\s*inout\s+vec3\s+\w+\s*\)/;
    c.programs.forEach((p, i) => {
      if (!SIG.test(p.source)) {
        this.errors.set(i, 'must declare void material(Surf s, float t, inout vec3 R, inout vec3 T, inout vec3 E)');
        bodies.push(`void material_${i}(Surf s, float t, inout vec3 R, inout vec3 T, inout vec3 E) { R = vec3(1.0, 0.0, 1.0); T = vec3(0.0); E = vec3(0.0); }`);
        return;
      }
      const renamed = p.source.replace(/void\s+material\s*\(/, `void material_${i}(`);
      const err = this.tryCompile(HEAD + `#line 1\n` + renamed + `\nvoid main() { o = vec4(0.0); }`);
      if (err) { this.errors.set(i, err); bodies.push(`void material_${i}(Surf s, float t, inout vec3 R, inout vec3 T, inout vec3 E) { R = vec3(1.0, 0.0, 1.0); T = vec3(0.0); E = vec3(0.0); }`); }
      else bodies.push(renamed);
    });
    const dispatch = c.programs.map((_, i) => `  ${i ? 'else ' : ''}if (m == ${i}) material_${i}(s, uTime, R, T, E);`).join('\n');
    const src = HEAD + bodies.join('\n') + `
void main() {
  ivec2 px = ivec2(gl_FragCoord.xy);
  int m = px.y / ${TILE};
  vec2 uv = (vec2(px.x % ${TILE}, px.y % ${TILE}) + 0.5) / float(${TILE}) * 2.0 - 1.0;
  vec3 c0 = texelFetch(uFrame, ivec2(0, m), 0).xyz, tu = texelFetch(uFrame, ivec2(1, m), 0).xyz, tv = texelFetch(uFrame, ivec2(2, m), 0).xyz;
  Surf s = Surf(uv, c0 + tu * uv.x + tv * uv.y, float(m));
  vec3 R = texelFetch(uFrame, ivec2(3, m), 0).xyz, T = texelFetch(uFrame, ivec2(4, m), 0).xyz, E = texelFetch(uFrame, ivec2(5, m), 0).xyz;
${dispatch}
  int tile = px.x / ${TILE};
  o = vec4(tile == 0 ? clamp(R, 0.0, 1.0) : tile == 1 ? clamp(T, 0.0, 1.0) : max(E, 0.0), 1.0);
}`;
    const err = this.tryCompile(src, true);
    if (err) this.errors.set(-1, err);
  }

  private tryCompile(fs: string, install = false): string | null {
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
      const v = mk(gl.VERTEX_SHADER, VS), f = mk(gl.FRAGMENT_SHADER, fs);
      const p = gl.createProgram()!;
      gl.attachShader(p, v); gl.attachShader(p, f); gl.linkProgram(p);
      if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p) || 'link error');
      if (install) {
        if (this.prog) gl.deleteProgram(this.prog);
        this.prog = p;
        for (const n of ['uFrame', 'uTime', 'uImage']) this.uni[n] = gl.getUniformLocation(p, n);
      } else gl.deleteProgram(p);
      return null;
    } catch (e) { return (e as Error).message; }
  }

  /** load a picture for uImage (sRGB-decoded to linear by the GPU) */
  setImage(img: HTMLImageElement | ImageBitmap) {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.image);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, 1);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.SRGB8_ALPHA8, gl.RGBA, gl.UNSIGNED_BYTE, img);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, 0);
  }

  run(t: number) {
    const gl = this.gl;
    if (!this.prog || !this.count) return;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    gl.viewport(0, 0, TILE * 3, TILE * this.count);
    gl.useProgram(this.prog);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.frames);
    gl.uniform1i(this.uni.uFrame, 0);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.image);
    gl.uniform1i(this.uni.uImage, 1);
    gl.uniform1f(this.uni.uTime, t);
    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.bindVertexArray(null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }
}
