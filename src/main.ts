import { Compiled, Material, PRESETS, Scene, Sel, V3, compile, mul, validate } from './scene';
import { LIBRARY } from './poly';
import { Tracer } from './tracer';
import { Orbit } from './camera';
import { Line, Overlay } from './overlay';
import { mouseRay, pick } from './pick';

// --- dom helpers -----------------------------------------------------------------------
const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
type Kid = Node | string;
function el<K extends keyof HTMLElementTagNameMap>(tag: K, attrs: Record<string, string> = {}, ...kids: Kid[]) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  e.append(...kids);
  return e;
}
function num(label: string, value: number, on: (v: number) => void, step = 1, min?: number, max?: number) {
  const i = el('input', { type: 'number', step: String(step), value: String(+value.toFixed(3)) });
  if (min !== undefined) i.min = String(min);
  if (max !== undefined) i.max = String(max);
  i.oninput = () => { const v = +i.value; if (isFinite(v)) on(v); };
  return el('label', { class: 'n' }, label, i);
}
function range(label: string, value: number, min: number, max: number, step: number, on: (v: number) => void) {
  const out = el('span', {}, value.toFixed(step < 1 ? 2 : 0));
  const i = el('input', { type: 'range', min: String(min), max: String(max), step: String(step), value: String(value) });
  i.oninput = () => { out.textContent = (+i.value).toFixed(step < 1 ? 2 : 0); on(+i.value); };
  return el('label', {}, label, out, i);
}
function select(opts: string[], value: string, on: (v: string) => void) {
  const s = el('select');
  for (const o of opts) s.add(new Option(o, o, false, o === value));
  s.onchange = () => on(s.value);
  return s;
}
function button(text: string, on: () => void, cls = '') {
  const b = el('button', cls ? { class: cls } : {}, text);
  b.onclick = on;
  return b;
}
const clamp01 = (v: number) => Math.min(1, Math.max(0, v));
const toHex = (c: V3) => '#' + c.map(v => Math.round(Math.pow(clamp01(v), 1 / 2.2) * 255).toString(16).padStart(2, '0')).join('');
const fromHex = (h: string): V3 => [1, 3, 5].map(i => Math.pow(parseInt(h.slice(i, i + 2), 16) / 255, 2.2)) as V3;
function color(label: string, value: V3, on: (v: V3) => void) {
  const i = el('input', { type: 'color', value: toHex(value) });
  i.oninput = () => on(fromHex(i.value));
  return el('label', { class: 'n' }, label, i);
}

// --- gpu + camera ------------------------------------------------------------------------
const canvas = $<HTMLCanvasElement>('view');
const status = $('status');
let tracer: Tracer, overlay: Overlay;
try { tracer = new Tracer(canvas); overlay = new Overlay(tracer.gl); }
catch (e) { status.textContent = `GPU: ${(e as Error).message}`; throw e; }
const orbit = new Orbit(canvas);
let renderScale = 1;
function fit() {
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  tracer.resize(Math.max(1, Math.round(canvas.clientWidth * dpr * renderScale)),
    Math.max(1, Math.round(canvas.clientHeight * dpr * renderScale)));
}

// --- state -------------------------------------------------------------------------------
let scene: Scene = PRESETS[0]();
let compiled: Compiled = compile(scene);
let sel: Sel | null = null;
let hover: Sel | null = null;
let wire = true;
const json = $<HTMLTextAreaElement>('json');
const panel = $('panel');
const same = (a: Sel | null, b: Sel | null) => !!a && !!b && a.kind === b.kind && a.i === b.i;
const extent = () => Math.max(...compiled.hull.verts.flat().map(Math.abs));

/** recompile after any edit: tracer, overlay, json */
function rebuild() {
  try { compiled = compile(scene); status.textContent = ''; }
  catch (e) { status.textContent = `geometry: ${(e as Error).message}`; return; }
  tracer.setScene(compiled);
  updateLines();
  if (document.activeElement !== json) json.value = JSON.stringify(scene, null, 2);
}

function updateLines() {
  const L: Line[] = [];
  const h = compiled.hull;
  const GRAY: V3 = [0.32, 0.32, 0.35], ACC: V3 = [1, 0.65, 0.2], HOV: V3 = [0.9, 0.9, 0.9], CUT: V3 = [0.2, 0.75, 0.9];
  const is = (s: Sel | null, kind: Sel['kind'], i: number) => !!s && s.kind === kind && s.i === i;
  h.edges.forEach((e, i) => {
    let c = GRAY;
    if (hover?.kind === 'face' && e.faces.includes(hover.i)) c = HOV;
    if (sel?.kind === 'face' && e.faces.includes(sel.i)) c = ACC;
    if (is(hover, 'edge', i)) c = HOV;
    if (is(sel, 'edge', i)) c = ACC;
    L.push({ a: h.verts[e.a], b: h.verts[e.b], color: c });
  });
  compiled.cutRings.forEach((ring, i) => {
    if (!ring) return;
    const c = is(sel, 'cut', i) ? ACC : is(hover, 'cut', i) ? HOV : CUT;
    ring.forEach((p, k) => L.push({ a: p, b: ring[(k + 1) % ring.length], color: c }));
  });
  compiled.caps.forEach(cp => {
    const m = Math.max(...cp.color) || 1;
    const c = same(cp.ref, sel) ? ACC : same(cp.ref, hover) ? HOV : mul(cp.color, 1 / m);
    L.push({ a: cp.a, b: cp.b, color: c });
  });
  overlay.set(L);
}

function load(s: Scene) {
  scene = s; sel = null; hover = null;
  orbit.setEye(s.eye.pos, s.eye.target, s.eye.fov);
  rebuild();
  renderPanel();
}
function setSel(s: Sel | null) { sel = s; updateLines(); renderPanel(); }

// --- panel ---------------------------------------------------------------------------------
function matEditor(get: () => Material, on: () => void) {
  const box = el('div', { class: 'mat' });
  const refresh = () => {
    const m = get();
    const kind = select(['open', 'matte', 'mirror'], m.kind, v => { get().kind = v as Material['kind']; refresh(); on(); });
    const rows: Kid[] = [kind];
    if (m.kind === 'mirror') rows.push(
      num('reflect', m.reflect ?? 0.95, v => { get().reflect = v; on(); }, 0.01, 0, 1),
      num('transmit', m.transmit ?? 0, v => { get().transmit = v; on(); }, 0.01, 0, 1));
    if (m.kind === 'matte') rows.push(num('albedo', m.albedo ?? 0.04, v => { get().albedo = v; on(); }, 0.01, 0, 1));
    box.replaceChildren(...rows);
  };
  refresh();
  return box;
}

function shellSection() {
  const sh = scene.shell;
  const st = sh.stretch ?? (sh.stretch = [1, 1, 1]);
  const ro = sh.rotate ?? (sh.rotate = [0, 0, 0]);
  const faceOpts = compiled.hull.faces.map((f, i) => `face ${i} · ${f.length}-gon · ${(sh.faces[i] ?? sh.defaultFace).kind}`);
  const faceSel = select(['select a face…', ...faceOpts], 'select a face…', v => {
    const i = faceOpts.indexOf(v); if (i >= 0) setSel({ kind: 'face', i });
  });
  return el('section', {},
    el('h2', {}, 'shell'),
    el('div', { class: 'grid2' },
      select([...LIBRARY, 'custom'], sh.poly, v => {
        sh.poly = v; sh.faces = {}; scene.edgeStrips = []; scene.cuts = scene.cuts.filter(() => true);
        if (v === 'custom' && !sh.vertices) sh.vertices = compiled.hull.verts.map(p => mul(p, 1 / sh.size));
        sel = null; rebuild(); renderPanel();
      }),
      num('size mm', sh.size, v => { sh.size = v; rebuild(); }, 5, 1)),
    el('div', { class: 'grid3' },
      num('stretch x', st[0], v => { st[0] = v; rebuild(); }, 0.05, 0.01),
      num('y', st[1], v => { st[1] = v; rebuild(); }, 0.05, 0.01),
      num('z', st[2], v => { st[2] = v; rebuild(); }, 0.05, 0.01)),
    el('div', { class: 'grid3' },
      num('rotate x°', ro[0], v => { ro[0] = v; rebuild(); }, 5),
      num('y°', ro[1], v => { ro[1] = v; rebuild(); }, 5),
      num('z°', ro[2], v => { ro[2] = v; rebuild(); }, 5)),
    el('div', { class: 'sub' }, 'default face'),
    matEditor(() => sh.defaultFace, rebuild),
    faceSel,
  );
}

function selectionSection() {
  const h = compiled.hull;
  const s = el('section', {});
  if (!sel) {
    s.append(el('h2', {}, 'selection'), el('p', { class: 'hint' }, 'click a face, edge, cut outline or strip in the view'));
    return s;
  }
  const sh = scene.shell;
  if (sel.kind === 'face') {
    const i = sel.i;
    const mat = () => sh.faces[i] ?? (sh.faces[i] = structuredClone(sh.defaultFace));
    const edges = h.edges.map((e, k) => e.faces.includes(i) ? k : -1).filter(k => k >= 0);
    s.append(el('h2', {}, `face ${i} · ${h.faces[i].length} sides`),
      matEditor(mat, rebuild),
      el('div', { class: 'row' },
        button('all faces like this', () => { sh.defaultFace = structuredClone(mat()); sh.faces = {}; rebuild(); renderPanel(); }),
        button('use default', () => { delete sh.faces[i]; rebuild(); renderPanel(); })),
      el('div', { class: 'row' },
        button('LEDs on its edges', () => {
          for (const e of edges) if (!scene.edgeStrips.some(x => x.edge === e)) scene.edgeStrips.push({ edge: e });
          rebuild();
        }),
        button('no LEDs on its edges', () => { scene.edgeStrips = scene.edgeStrips.filter(x => !edges.includes(x.edge)); rebuild(); })));
  } else if (sel.kind === 'edge') {
    const i = sel.i, e = h.edges[i];
    const es = scene.edgeStrips.find(x => x.edge === i);
    const D = scene.edgeDefaults;
    const cb = el('input', { type: 'checkbox' });
    cb.checked = !!es;
    cb.onchange = () => {
      if (cb.checked) scene.edgeStrips.push({ edge: i });
      else scene.edgeStrips = scene.edgeStrips.filter(x => x.edge !== i);
      rebuild(); renderPanel();
    };
    s.append(el('h2', {}, `edge ${i} · faces ${e.faces[0]} & ${e.faces[1]}`), el('label', { class: 'cb' }, cb, ' LED strip on this edge'));
    if (es) s.append(
      el('div', { class: 'sub' }, 'overrides (blank = default)'),
      color('color', es.color ?? D.color, v => { es.color = v; rebuild(); }),
      el('div', { class: 'grid2' },
        num('radiance', es.radiance ?? D.radiance, v => { es.radiance = v; rebuild(); }, 1, 0),
        num('inset mm', es.inset ?? D.inset, v => { es.inset = v; rebuild(); }, 1),
        num('pitch mm', es.pitch ?? D.pitch, v => { es.pitch = v; rebuild(); }, 0.5, 0),
        num('radius mm', es.radius ?? D.radius, v => { es.radius = v; rebuild(); }, 0.25, 0.1)),
      button('clear overrides', () => { scene.edgeStrips[scene.edgeStrips.indexOf(es)] = { edge: i }; rebuild(); renderPanel(); }));
  } else if (sel.kind === 'cut') {
    const i = sel.i, c = scene.cuts[i];
    const n = c.normal;
    const L = Math.hypot(...n) || 1;
    let yaw = Math.atan2(n[0], n[2]) * 180 / Math.PI, pitch = Math.asin(Math.max(-1, Math.min(1, n[1] / L))) * 180 / Math.PI;
    const setN = () => {
      const y = yaw * Math.PI / 180, p = pitch * Math.PI / 180;
      c.normal = [Math.sin(y) * Math.cos(p), Math.sin(p), Math.cos(y) * Math.cos(p)];
      rebuild();
    };
    const R = extent() * 1.05;
    const name = el('input', { type: 'text', value: c.name ?? '' });
    name.oninput = () => { c.name = name.value; rebuild(); };
    s.append(el('h2', {}, `cut ${i}`), name,
      range('offset mm', c.offset, -R, R, 0.5, v => { c.offset = v; rebuild(); }),
      range('yaw°', yaw, -180, 180, 0.5, v => { yaw = v; setN(); }),
      range('pitch°', pitch, -89, 89, 0.5, v => { pitch = v; setN(); }),
      range('scale', c.scale ?? 1, 0.05, 1, 0.01, v => { c.scale = v; rebuild(); }),
      matEditor(() => c.material, rebuild),
      button('delete cut', () => { scene.cuts.splice(i, 1); setSel(null); rebuild(); }, 'danger'));
  } else {
    const i = sel.i, t = scene.strips[i];
    const name = el('input', { type: 'text', value: t.name ?? '' });
    name.oninput = () => { t.name = name.value; rebuild(); };
    const v3 = (label: string, v: V3) => el('div', { class: 'grid3' },
      num(label + ' x', v[0], x => { v[0] = x; rebuild(); }), num('y', v[1], x => { v[1] = x; rebuild(); }), num('z', v[2], x => { v[2] = x; rebuild(); }));
    s.append(el('h2', {}, `strip ${i}`), name, v3('a', t.a), v3('b', t.b),
      color('color', t.color, v => { t.color = v; rebuild(); }),
      el('div', { class: 'grid3' },
        num('radiance', t.radiance, v => { t.radiance = v; rebuild(); }, 1, 0),
        num('pitch', t.pitch, v => { t.pitch = v; rebuild(); }, 0.5, 0),
        num('radius', t.radius, v => { t.radius = v; rebuild(); }, 0.25, 0.1)),
      button('delete strip', () => { scene.strips.splice(i, 1); setSel(null); rebuild(); }, 'danger'));
  }
  return s;
}

function actionsSection() {
  const D = scene.edgeDefaults;
  return el('section', {},
    el('h2', {}, 'add'),
    el('div', { class: 'row' },
      button('cut plane', () => {
        scene.cuts.push({ name: `cut ${scene.cuts.length}`, normal: [0, 0, 1], offset: 0, scale: 1,
          material: { kind: 'mirror', reflect: 0.95, transmit: 0 } });
        rebuild(); setSel({ kind: 'cut', i: scene.cuts.length - 1 });
      }),
      button('free strip', () => {
        const R = extent(), z = Math.min(...compiled.hull.verts.map(v => v[2])) + 5;
        scene.strips.push({ name: `strip ${scene.strips.length}`, a: [-R / 2, 0, z], b: [R / 2, 0, z],
          color: [...D.color] as V3, radiance: D.radiance, pitch: D.pitch, radius: D.radius });
        rebuild(); setSel({ kind: 'strip', i: scene.strips.length - 1 });
      })),
    el('div', { class: 'row' },
      button('LEDs on all edges', () => { scene.edgeStrips = compiled.hull.edges.map((_, edge) => scene.edgeStrips.find(x => x.edge === edge) ?? { edge }); rebuild(); }),
      button('clear all LEDs', () => { scene.edgeStrips = []; rebuild(); })),
    el('h2', {}, 'edge strip defaults'),
    color('color', D.color, v => { D.color = v; rebuild(); }),
    el('div', { class: 'grid2' },
      num('radiance', D.radiance, v => { D.radiance = v; rebuild(); }, 1, 0),
      num('inset mm', D.inset, v => { D.inset = v; rebuild(); }, 1),
      num('pitch mm', D.pitch, v => { D.pitch = v; rebuild(); }, 0.5, 0),
      num('radius mm', D.radius, v => { D.radius = v; rebuild(); }, 0.25, 0.1)));
}

function renderPanel() { panel.replaceChildren(shellSection(), selectionSection(), actionsSection()); }

// --- picking ---------------------------------------------------------------------------------
let down: { x: number; y: number } | null = null;
function pickAt(x: number, y: number): Sel | null {
  const r = canvas.getBoundingClientRect();
  const cam = orbit.basis();
  const { ro, rd } = mouseRay(cam, x - r.left, y - r.top, r.width, r.height);
  return pick(compiled, ro, rd, 2 * cam.tanHalf / r.height);
}
canvas.addEventListener('pointerdown', e => { down = { x: e.clientX, y: e.clientY }; });
canvas.addEventListener('pointerup', e => {
  if (down && Math.hypot(e.clientX - down.x, e.clientY - down.y) < 4 && e.button === 0) setSel(pickAt(e.clientX, e.clientY));
  down = null;
});
canvas.addEventListener('pointermove', e => {
  if (e.buttons) return;
  const h = pickAt(e.clientX, e.clientY);
  if (!same(h, hover) && (h || hover)) { hover = h; updateLines(); }
});
canvas.addEventListener('pointerleave', () => { if (hover) { hover = null; updateLines(); } });
window.addEventListener('keydown', e => {
  if ((e.target as HTMLElement).matches('input, textarea, select')) return;
  if (e.key === 'Escape') setSel(null);
  if ((e.key === 'Delete' || e.key === 'Backspace') && sel) {
    if (sel.kind === 'cut') scene.cuts.splice(sel.i, 1);
    else if (sel.kind === 'strip') scene.strips.splice(sel.i, 1);
    else if (sel.kind === 'edge') scene.edgeStrips = scene.edgeStrips.filter(x => x.edge !== (sel as Sel).i);
    else return;
    setSel(null); rebuild();
  }
});

// --- global controls --------------------------------------------------------------------------
const presetSel = $<HTMLSelectElement>('preset');
PRESETS.forEach((p, i) => presetSel.add(new Option(p().name, String(i))));
presetSel.onchange = () => load(PRESETS[+presetSel.value]());
$('apply').onclick = () => {
  try { load(validate(JSON.parse(json.value))); }
  catch (e) { status.textContent = `scene: ${(e as Error).message}`; }
};
$('save').onclick = () => {
  scene.eye = { pos: orbit.eye(), target: [...orbit.target] as V3, fov: orbit.fov };
  const blob = new Blob([JSON.stringify(scene, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `${scene.name.replace(/[^a-z0-9]+/gi, '-')}.json`;
  a.click();
};
$<HTMLInputElement>('open').onchange = async e => {
  const f = (e.target as HTMLInputElement).files?.[0];
  if (!f) return;
  try { load(validate(JSON.parse(await f.text()))); }
  catch (err) { status.textContent = `scene: ${(err as Error).message}`; }
};
$('reset-eye').onclick = () => { orbit.setEye(scene.eye.pos, scene.eye.target, scene.eye.fov); tracer.reset(); };
$('menu-btn').onclick = () => document.body.classList.toggle('open');
const bounceIn = $<HTMLInputElement>('bounces'), expIn = $<HTMLInputElement>('exposure'), scaleIn = $<HTMLInputElement>('scale');
bounceIn.oninput = () => { tracer.maxBounce = +bounceIn.value; $('bounces-v').textContent = bounceIn.value; tracer.reset(); };
expIn.oninput = () => { tracer.exposure = Math.pow(2, +expIn.value); $('exposure-v').textContent = `${(+expIn.value).toFixed(1)} EV`; };
scaleIn.oninput = () => { renderScale = +scaleIn.value; $('scale-v').textContent = `${renderScale}×`; fit(); };
$<HTMLInputElement>('wire').onchange = e => { wire = (e.target as HTMLInputElement).checked; };
orbit.onChange = () => tracer.reset();
bounceIn.oninput(new Event('input')); expIn.oninput(new Event('input'));

new ResizeObserver(fit).observe(canvas);
load(PRESETS[0]());

// --- frame loop: adapt samples per frame to ~16 ms, stop tracing at a sample cap ---------------
const CAP = 8192;
let last = performance.now(), spp = 1, ema = 16;
function tick(now: number) {
  const dt = now - last; last = now;
  const cam = orbit.basis();
  if (tracer.frame < CAP) {
    ema = ema * 0.9 + dt * 0.1;
    if (ema < 12 && spp < 16) spp++; else if (ema > 24 && spp > 1) spp--;
  }
  tracer.render(cam, tracer.frame < CAP ? spp : 0);
  if (wire) overlay.draw(cam, canvas.width, canvas.height);
  $('samples').textContent = `${tracer.frame} spp · ${spp}/frame · ${ema.toFixed(0)} ms · ${canvas.width}×${canvas.height}`;
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);

// debug handles for the console
Object.defineProperties(window, {
  tracer: { value: tracer }, orbit: { value: orbit }, fit: { value: fit }, load: { value: load },
  scene: { get: () => scene }, compiled: { get: () => compiled },
});
