// Orbit camera: yaw/pitch around a target at a distance. Drag orbits, wheel zooms,
// shift-drag or right-drag pans, two fingers pan and pinch on touch.

import { V3, add, mul, sub, len } from './scene';
import { CameraBasis, basis } from './tracer';

export class Orbit {
  target: V3 = [0, 0, 0];
  dist = 700;
  yaw = 0;     // radians about +y, 0 looks down -z... see eye()
  pitch = 0;   // radians above the xz plane
  fov = 32;
  onChange: () => void = () => {};

  constructor(private el: HTMLElement) {
    let drag: { x: number; y: number; btn: number; shift: boolean } | null = null;
    el.addEventListener('pointerdown', e => {
      drag = { x: e.clientX, y: e.clientY, btn: e.button, shift: e.shiftKey };
      el.setPointerCapture(e.pointerId);
    });
    el.addEventListener('pointermove', e => {
      if (!drag) return;
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      drag.x = e.clientX; drag.y = e.clientY;
      if (drag.btn === 2 || drag.shift) this.pan(dx, dy);
      else { this.yaw -= dx * 0.005; this.pitch = Math.max(-1.5, Math.min(1.5, this.pitch + dy * 0.005)); }
      this.onChange();
    });
    const end = () => { drag = null; };
    el.addEventListener('pointerup', end); el.addEventListener('pointercancel', end);
    el.addEventListener('contextmenu', e => e.preventDefault());
    el.addEventListener('wheel', e => {
      e.preventDefault();
      this.dist *= Math.exp(e.deltaY * 0.001);
      this.dist = Math.max(10, this.dist);
      this.onChange();
    }, { passive: false });
  }

  /** set from a scene eye definition */
  setEye(pos: V3, target: V3, fov: number) {
    this.target = [...target] as V3;
    const d = sub(pos, target);
    this.dist = len(d);
    this.yaw = Math.atan2(d[0], d[2]);
    this.pitch = Math.asin(Math.max(-1, Math.min(1, d[1] / (this.dist || 1))));
    this.fov = fov;
  }

  eye(): V3 {
    const cp = Math.cos(this.pitch);
    return add(this.target, mul([Math.sin(this.yaw) * cp, Math.sin(this.pitch), Math.cos(this.yaw) * cp], this.dist));
  }

  basis(): CameraBasis { return basis(this.eye(), this.target, this.fov); }

  private pan(dx: number, dy: number) {
    const b = this.basis();
    const s = this.dist * Math.tan(this.fov * Math.PI / 360) * 2 / this.el.clientHeight;
    this.target = add(this.target, add(mul(b.right, -dx * s), mul(b.up, dy * s)));
  }
}
