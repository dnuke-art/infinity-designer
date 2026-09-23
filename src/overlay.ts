// Wireframe overlay drawn on top of the traced image: shell edges, cut outlines, strips,
// with the selected and hovered items in accent colours. Same camera model as the tracer
// (a point projects to the pixel whose primary ray passes through it) so the lines sit
// exactly on the physical geometry.

import { V3 } from './scene';
import { CameraBasis } from './tracer';

const VS = `#version 300 es
in vec3 aPos; in vec3 aCol;
uniform vec3 uEye, uFwd, uRight, uUp;
uniform float uTanHalf, uAspect;
out vec3 vCol;
void main() {
  vec3 q = aPos - uEye;
  float z = dot(q, uFwd);
  gl_Position = vec4(dot(q, uRight) / (uTanHalf * uAspect), dot(q, uUp) / uTanHalf, 0.0, z);
  vCol = aCol;
}`;
const FS = `#version 300 es
precision mediump float;
in vec3 vCol; out vec4 o;
void main() { o = vec4(vCol, 1.0); }`;

export interface Line { a: V3; b: V3; color: V3 }

export class Overlay {
  private prog: WebGLProgram;
  private vao: WebGLVertexArrayObject;
  private buf: WebGLBuffer;
  private count = 0;
  private uni: Record<string, WebGLUniformLocation | null> = {};

  constructor(private gl: WebGL2RenderingContext) {
    const mk = (type: number, src: string) => {
      const sh = gl.createShader(type)!; gl.shaderSource(sh, src); gl.compileShader(sh);
      if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(sh) || 'overlay shader');
      return sh;
    };
    this.prog = gl.createProgram()!;
    gl.attachShader(this.prog, mk(gl.VERTEX_SHADER, VS));
    gl.attachShader(this.prog, mk(gl.FRAGMENT_SHADER, FS));
    gl.linkProgram(this.prog);
    for (const n of ['uEye', 'uFwd', 'uRight', 'uUp', 'uTanHalf', 'uAspect']) this.uni[n] = gl.getUniformLocation(this.prog, n);
    this.vao = gl.createVertexArray()!;
    this.buf = gl.createBuffer()!;
    gl.bindVertexArray(this.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buf);
    const aPos = gl.getAttribLocation(this.prog, 'aPos'), aCol = gl.getAttribLocation(this.prog, 'aCol');
    gl.enableVertexAttribArray(aPos); gl.vertexAttribPointer(aPos, 3, gl.FLOAT, false, 24, 0);
    gl.enableVertexAttribArray(aCol); gl.vertexAttribPointer(aCol, 3, gl.FLOAT, false, 24, 12);
    gl.bindVertexArray(null);
  }

  set(lines: Line[]) {
    const data = new Float32Array(lines.length * 12);
    lines.forEach((l, i) => data.set([...l.a, ...l.color, ...l.b, ...l.color], i * 12));
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buf);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW);
    this.count = lines.length * 2;
  }

  draw(cam: CameraBasis, w: number, h: number) {
    if (!this.count) return;
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, w, h);
    gl.useProgram(this.prog);
    gl.uniform3fv(this.uni.uEye, cam.eye); gl.uniform3fv(this.uni.uFwd, cam.fwd);
    gl.uniform3fv(this.uni.uRight, cam.right); gl.uniform3fv(this.uni.uUp, cam.up);
    gl.uniform1f(this.uni.uTanHalf, cam.tanHalf); gl.uniform1f(this.uni.uAspect, w / h);
    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.LINES, 0, this.count);
    gl.bindVertexArray(null);
  }
}
