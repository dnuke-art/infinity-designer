import { Compiled, Laser, Material, PRESETS, Scene, Sel, V3, compile, dot, ledMap, mul, sub, validate } from './scene';
import { PATTERNS, PATTERN_API, PatternPass } from './leds';
import { MATERIALS, MATERIAL_API, MaterialPass } from './mats';
import { LIBRARY, parseSTL } from './poly';
import { Tracer } from './tracer';
import { Orbit } from './camera';
import { Line, Overlay } from './overlay';
import { mouseRay, pick } from './pick';
import { blenderExport } from './export';

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
let tracer: Tracer, overlay: Overlay, leds: PatternPass, mats: MaterialPass;
try {
  tracer = new Tracer(canvas); overlay = new Overlay(tracer.gl);
  leds = new PatternPass(tracer.gl); tracer.ledTex = leds.tex;
  mats = new MaterialPass(tracer.gl); tracer.matTex = mats.tex;
}
catch (e) { status.textContent = `GPU: ${(e as Error).message}`; throw e; }
const orbit = new Orbit(canvas);
let renderScale = 1;
/** an offline render in progress: the buffers are at its size until it finishes */
let job: { w: number; h: number; target: number; format: 'jpg' | 'png'; quality: number } | null = null;
function fit() {
  if (job) return;
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
let playing = false, time = 0, speed = 1;
let patternErr = '';
let scrub: { input: HTMLInputElement; out: HTMLElement } | null = null;
const json = $<HTMLTextAreaElement>('json');
const panel = $('panel');
const bounceIn = $<HTMLInputElement>('bounces'), expIn = $<HTMLInputElement>('exposure'), scaleIn = $<HTMLInputElement>('scale');
const same = (a: Sel | null, b: Sel | null) => !!a && !!b && a.kind === b.kind && a.i === b.i;
const selName = (s: Sel) => `${s.kind} ${s.i}`;
const DEG = Math.PI / 180;
const extent = () => Math.max(...compiled.hull.verts.flat().map(Math.abs));

/** recompile after any edit: tracer, overlay, json */
function rebuild() {
  try { compiled = compile(scene); status.textContent = ''; }
  catch (e) { status.textContent = `geometry: ${(e as Error).message}`; return; }
  tracer.setScene(compiled);
  leds.setScene(compiled);
  mats.setScene(compiled);
  updateFog();
  if (mats.errors.size) {
    const msgs = [...mats.errors].map(([i, e]) => (i < 0 ? 'materials: ' : `${selName(mats.refs[i])} program: `) + e);
    status.textContent = msgs.join('\n');
  }
  updateLines();
  if (document.activeElement !== json) json.value = JSON.stringify(scene, null, 2);
}

/** hand the fog to the tracer, with whether the eye sits inside the shell */
function updateFog() {
  const f = scene.fog;
  const eye = orbit.eye();
  const h = compiled.hull;
  const inside = h.faces.every((face, i) => dot(h.normals[i], sub(eye, h.verts[face[0]])) <= 0);
  tracer.fog = f ? { sigmaT: f.density / 1000, albedo: f.albedo, g: f.g, eyeInside: inside }
    : { sigmaT: 0, albedo: [1, 1, 1], g: 0, eyeInside: false };
}

/** compile the scene's pattern (or the solid default); remembers the error for the panel */
function applyPattern() {
  const src = scene.pattern?.source ?? PATTERNS[0].source;
  patternErr = leds.setSource(src) ?? '';
  tracer.reset();
}

function updateLines() {
  const L: Line[] = [];
  const h = compiled.hull;
  const GRAY: V3 = [0.32, 0.32, 0.35], ACC: V3 = [1, 0.65, 0.2], HOV: V3 = [0.9, 0.9, 0.9], CUT: V3 = [0.2, 0.75, 0.9], ANIM: V3 = [0.9, 0.3, 0.9];
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
    const animated = compiled.programs.some(p => p.ref.kind === 'cut' && p.ref.i === i);
    const c = is(sel, 'cut', i) ? ACC : is(hover, 'cut', i) ? HOV : animated ? ANIM : CUT;
    ring.forEach((p, k) => L.push({ a: p, b: ring[(k + 1) % ring.length], color: c }));
  });
  compiled.caps.forEach(cp => {
    if (cp.ref.kind === 'laser') return;
    const m = Math.max(...cp.color) || 1;
    const c = same(cp.ref, sel) ? ACC : same(cp.ref, hover) ? HOV : mul(cp.color, 1 / m);
    L.push({ a: cp.a, b: cp.b, color: c });
  });
  compiled.beams.forEach(b => {
    const ref: Sel = { kind: 'laser', i: b.laser };
    const m = Math.max(...b.color) || 1;
    const c = same(ref, sel) ? ACC : same(ref, hover) ? HOV : mul(b.color, 0.6 / m);
    L.push({ a: b.a, b: b.b, color: c });
  });
  overlay.set(L);
}

function load(s: Scene) {
  scene = s; sel = null; hover = null;
  orbit.setEye(s.eye.pos, s.eye.target, s.eye.fov);
  expIn.value = String(s.eye.exposure ?? -2); expIn.oninput!(new Event('input'));
  rebuild();
  applyPattern();
  renderPanel();
}
function setSel(s: Sel | null) { sel = s; updateLines(); renderPanel(); }

// --- panel ---------------------------------------------------------------------------------
/** a reflect/transmit field: a number, or three when the material is tinted */
function rtField(label: string, m: Material, key: 'reflect' | 'transmit', dflt: number, on: () => void) {
  const v = m[key] ?? dflt;
  if (typeof v === 'number') {
    const row = el('div', { class: 'row' },
      num(label, v, x => { m[key] = x; on(); }, 0.01, 0, 1),
      button('tint', () => { m[key] = [v, v, v]; on(); rerenderPanelSoon(); }, 'small'));
    return row;
  }
  const tri = v as V3;
  return el('div', { class: 'grid3' },
    num(label + ' r', tri[0], x => { tri[0] = x; on(); }, 0.01, 0, 1),
    num('g', tri[1], x => { tri[1] = x; on(); }, 0.01, 0, 1),
    num('b', tri[2], x => { tri[2] = x; on(); }, 0.01, 0, 1));
}
/** pitch in mm and LEDs per metre, kept in step */
function pitchFields(get: () => number, set: (v: number) => void) {
  const per = num('LEDs / m', get() > 0 ? 1000 / get() : 0, v => { if (v > 0) { set(1000 / v); sync(); } }, 1, 0);
  const mm = num('pitch mm', get(), v => { set(v); sync(); }, 0.5, 0);
  const sync = () => {
    const p = get();
    per.querySelector('input')!.value = String(+(p > 0 ? 1000 / p : 0).toFixed(1));
    mm.querySelector('input')!.value = String(+p.toFixed(2));
  };
  return [per, mm];
}
let panelTimer = 0;
function rerenderPanelSoon() { clearTimeout(panelTimer); panelTimer = window.setTimeout(renderPanel, 0); }

function matEditor(get: () => Material, on: () => void) {
  const box = el('div', { class: 'mat' });
  const refresh = () => {
    const m = get();
    const kind = select(['open', 'matte', 'mirror', 'screen'], m.kind, v => { get().kind = v as Material['kind']; refresh(); on(); });
    const rows: Kid[] = [kind];
    if (m.kind === 'mirror' || m.kind === 'screen') {
      const screen = m.kind === 'screen';
      rows.push(rtField('reflect', m, 'reflect', screen ? 0.05 : 0.95, on), rtField('transmit', m, 'transmit', 0, on));
      if (screen) {
        const e = m.emit ?? 10;
        const tri: V3 = typeof e === 'number' ? [e, e, e] : e;
        const bright = Math.max(...tri) || 1;
        rows.push(el('div', { class: 'row' },
          color('emit', mul(tri, 1 / bright), c => { const mm = get(); const b = Math.max(...(Array.isArray(mm.emit) ? mm.emit : [bright])) || 1; mm.emit = mul(c, b); on(); }),
          num('radiance', bright, v => { const mm = get(); const cur = Array.isArray(mm.emit) ? mm.emit : [1, 1, 1] as V3; const mx = Math.max(...cur) || 1; mm.emit = mul(cur, v / mx); on(); }, 1, 0)));
      }
      // animated program
      const names = MATERIALS.map(x => x.name);
      const isPreset = MATERIALS.some(x => x.name === m.programName && x.source === m.program);
      const cur = m.program ? (isPreset ? m.programName! : 'custom') : 'static';
      const opts = ['static', ...(cur === 'custom' ? ['custom'] : []), ...names];
      const progSel = select(opts, cur, v => {
        const p = MATERIALS.find(x => x.name === v);
        const mm = get();
        if (p) { mm.program = p.source; mm.programName = p.name; }
        else if (v === 'static') { delete mm.program; delete mm.programName; }
        refresh(); on();
      });
      rows.push(el('label', {}, 'animate', progSel));
      if (m.program) {
        const src = el('textarea', { spellcheck: 'false', class: 'code' }, m.program);
        const apply = button('apply program', () => { const mm = get(); mm.program = src.value; mm.programName = 'custom'; on(); refresh(); }, 'primary');
        rows.push(el('details', {}, el('summary', {}, 'GLSL'), el('pre', { class: 'api' }, MATERIAL_API), src, apply),
          el('p', { class: 'hint' }, 'the program starts from the static values above and overrides what it sets'));
      }
    }
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
  const stlIn = el('input', { type: 'file', accept: '.stl', hidden: '' });
  stlIn.onchange = async () => {
    const f = stlIn.files?.[0];
    if (!f) return;
    try {
      const m = parseSTL(await f.arrayBuffer());
      sh.poly = 'mesh'; sh.mesh = m; sh.faces = {}; scene.edgeStrips = []; sel = null;
      status.textContent = '';
      rebuild(); renderPanel();
      status.textContent = `${f.name}: ${m.faces.length} triangles, ${compiled.hull.edges.length} edges`;
    } catch (e) { status.textContent = `stl: ${(e as Error).message}`; }
  };
  return el('section', {},
    el('h2', {}, 'shell'),
    el('div', { class: 'grid2' },
      select([...LIBRARY, 'custom', ...(sh.mesh ? ['mesh'] : [])], sh.poly, v => {
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
    el('label', { class: 'btn' }, 'import STL as the shell', stlIn),
  );
}

/** "LEDs 40–52 (13)" for the strip that a selection compiles to */
function ledRange(sel: Sel) {
  const cp = compiled.caps.find(c => same(c.ref, sel));
  if (!cp) return '';
  return cp.count > 1 ? `LEDs ${cp.base}–${cp.base + cp.count - 1} (${cp.count})` : `LED ${cp.base}`;
}
function reverseBox(get: () => boolean, set: (v: boolean) => void) {
  const cb = el('input', { type: 'checkbox' });
  cb.checked = get();
  cb.onchange = () => { set(cb.checked); rebuild(); renderPanel(); };
  return el('label', { class: 'cb' }, cb, ' reverse data direction');
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
      el('div', { class: 'sub' }, ledRange(sel)),
      reverseBox(() => !!es.reverse, v => { es.reverse = v || undefined; }),
      el('div', { class: 'sub' }, 'overrides (blank = default)'),
      color('color', es.color ?? D.color, v => { es.color = v; rebuild(); }),
      el('div', { class: 'grid2' },
        num('radiance', es.radiance ?? D.radiance, v => { es.radiance = v; rebuild(); }, 1, 0),
        num('inset mm', es.inset ?? D.inset, v => { es.inset = v; rebuild(); }, 1),
        ...pitchFields(() => es.pitch ?? D.pitch, v => { es.pitch = v; rebuild(); }),
        num('radius mm', es.radius ?? D.radius, v => { es.radius = v; rebuild(); }, 0.25, 0.1),
        num('cone °', es.cone ?? D.cone ?? 180, v => { es.cone = v; rebuild(); }, 5, 10, 180),
        num('shade lip mm', es.shade ?? D.shade ?? 0, v => { es.shade = v; rebuild(); }, 1, 0),
        num('tube radius mm', es.tube ?? D.tube ?? 0, v => { es.tube = v; rebuild(); }, 0.5, 0),
        num('blur mm', es.blur ?? D.blur ?? ((es.tube ?? D.tube ?? 0) * 1.5), v => { es.blur = v; rebuild(); }, 0.5, 0.1)),
      (() => {
        const auto = 'auto (viewing face, or both if equal)';
        const opts = [auto, `face ${e.faces[0]}`, `face ${e.faces[1]}`, 'both faces'];
        const cur = es.shadeFace === 'both' ? opts[3] : es.shadeFace === e.faces[0] ? opts[1] : es.shadeFace === e.faces[1] ? opts[2] : auto;
        return el('label', {}, 'lip on', select(opts, cur, v => {
          es.shadeFace = v === opts[1] ? e.faces[0] : v === opts[2] ? e.faces[1] : v === opts[3] ? 'both' : undefined; rebuild();
        }));
      })(),
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
  } else if (sel.kind === 'laser') {
    const i = sel.i, l = (scene.lasers ?? [])[i];
    if (!l) { setSel(null); return s; }
    const name = el('input', { type: 'text', value: l.name ?? '' });
    name.oninput = () => { l.name = name.value; rebuild(); };
    const d = l.dir, Ld = Math.hypot(...d) || 1;
    let yaw = Math.atan2(d[0], d[2]) / DEG, pitch = Math.asin(Math.max(-1, Math.min(1, d[1] / Ld))) / DEG;
    const setDir = () => { l.dir = [Math.sin(yaw * DEG) * Math.cos(pitch * DEG), Math.sin(pitch * DEG), Math.cos(yaw * DEG) * Math.cos(pitch * DEG)]; rebuild(); };
    const pos = el('div', { class: 'grid3' },
      num('pos x', l.pos[0], v => { l.pos[0] = v; rebuild(); }), num('y', l.pos[1], v => { l.pos[1] = v; rebuild(); }), num('z', l.pos[2], v => { l.pos[2] = v; rebuild(); }));
    s.append(el('h2', {}, `laser ${i} · ${compiled.beams.filter(b => b.laser === i).length} beam segments`), name, pos,
      range('yaw°', yaw, -180, 180, 0.5, v => { yaw = v; setDir(); }),
      range('pitch°', pitch, -89, 89, 0.5, v => { pitch = v; setDir(); }),
      range('fan°', l.fan, 0, 120, 1, v => { l.fan = v; rebuild(); }),
      range('roll°', l.roll ?? 0, -90, 90, 1, v => { l.roll = v; rebuild(); }),
      color('colour', l.color, v => { l.color = v; rebuild(); }),
      el('div', { class: 'grid3' },
        num('power', l.power, v => { l.power = v; rebuild(); }, 1, 0),
        num('radius', l.radius ?? 1, v => { l.radius = v; rebuild(); }, 0.25, 0.1),
        num('rays', l.rays ?? 24, v => { l.rays = Math.max(2, Math.round(v)); rebuild(); }, 1, 2, 64)),
      (() => {
        const cb = el('input', { type: 'checkbox' });
        cb.checked = !!l.discrete;
        cb.onchange = () => { l.discrete = cb.checked || undefined; rebuild(); renderPanel(); };
        return el('label', { class: 'cb' }, cb, ' grating: separate beams (dots) instead of a line');
      })(),
      el('p', { class: 'hint' }, 'power is a display unit: the spot radiance on a white matte surface. Animated mirrors reflect lasers with their static values.'),
      button('delete laser', () => { scene.lasers!.splice(i, 1); setSel(null); rebuild(); }, 'danger'));
  } else {
    const i = sel.i, t = scene.strips[i];
    const name = el('input', { type: 'text', value: t.name ?? '' });
    name.oninput = () => { t.name = name.value; rebuild(); };
    const v3 = (label: string, v: V3) => el('div', { class: 'grid3' },
      num(label + ' x', v[0], x => { v[0] = x; rebuild(); }), num('y', v[1], x => { v[1] = x; rebuild(); }), num('z', v[2], x => { v[2] = x; rebuild(); }));
    s.append(el('h2', {}, `strip ${i}`), name, el('div', { class: 'sub' }, ledRange(sel)),
      reverseBox(() => !!t.reverse, v => { t.reverse = v || undefined; }),
      v3('a', t.a), v3('b', t.b),
      color('color', t.color, v => { t.color = v; rebuild(); }),
      el('div', { class: 'grid2' },
        num('radiance', t.radiance, v => { t.radiance = v; rebuild(); }, 1, 0),
        num('radius', t.radius, v => { t.radius = v; rebuild(); }, 0.25, 0.1),
        ...pitchFields(() => t.pitch, v => { t.pitch = v; rebuild(); }),
        num('tube radius mm', t.tube ?? 0, v => { t.tube = v; rebuild(); }, 0.5, 0),
        num('blur mm', t.blur ?? ((t.tube ?? 0) * 1.5), v => { t.blur = v; rebuild(); }, 0.5, 0.1)),
      (() => {
        const cb = el('input', { type: 'checkbox' });
        cb.checked = !!t.normal;
        cb.onchange = () => { if (cb.checked) { t.normal = [0, 0, 1]; t.cone ??= 120; } else { delete t.normal; } rebuild(); renderPanel(); };
        return el('label', { class: 'cb' }, cb, ' directional (cone about an axis)');
      })(),
      ...(t.normal ? [v3('axis', t.normal), num('cone °', t.cone ?? 120, v => { t.cone = v; rebuild(); }, 5, 10, 180)] : []),
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
    el('div', { class: 'row' },
      button('laser', () => {
        const R = extent();
        const l: Laser = { name: `laser ${(scene.lasers ?? []).length}`, pos: [-R * 0.9, -R * 0.8, 0], dir: [1, 0.6, -0.3],
          color: [0.2, 1, 0.3], power: 40, fan: 40, roll: 0, radius: 1, rays: 20 };
        (scene.lasers ??= []).push(l);
        rebuild(); setSel({ kind: 'laser', i: scene.lasers!.length - 1 });
      })),
    el('h2', {}, 'fog'),
    (() => {
      const cb = el('input', { type: 'checkbox' });
      cb.checked = !!scene.fog;
      cb.onchange = () => { scene.fog = cb.checked ? { density: 3, albedo: [0.9, 0.9, 0.9], g: 0.5 } : undefined; rebuild(); renderPanel(); };
      return el('label', { class: 'cb' }, cb, ' fog inside the shell');
    })(),
    ...(scene.fog ? [
      range('density /m', scene.fog.density, 0.2, 20, 0.1, v => { scene.fog!.density = v; rebuild(); }),
      range('albedo', scene.fog.albedo[0], 0, 1, 0.01, v => { scene.fog!.albedo = [v, v, v]; rebuild(); }),
      range('forward scatter g', scene.fog.g, -0.5, 0.95, 0.01, v => { scene.fog!.g = v; rebuild(); }),
      el('p', { class: 'hint' }, `fog is lit by LEDs, laser lines and ${compiled.images.length} mirror images of them; deeper images arrive by chance only`),
    ] : []),
    el('h2', {}, 'edge strip defaults'),
    color('color', D.color, v => { D.color = v; rebuild(); }),
    el('div', { class: 'grid2' },
      num('radiance', D.radiance, v => { D.radiance = v; rebuild(); }, 1, 0),
      num('inset mm', D.inset, v => { D.inset = v; rebuild(); }, 1),
      ...pitchFields(() => D.pitch, v => { D.pitch = v; rebuild(); }),
      num('radius mm', D.radius, v => { D.radius = v; rebuild(); }, 0.25, 0.1),
      num('cone °', D.cone ?? 180, v => { D.cone = v; rebuild(); }, 5, 10, 180),
      num('shade lip mm', D.shade ?? 0, v => { D.shade = v; rebuild(); }, 1, 0),
      num('tube radius mm', D.tube ?? 0, v => { D.tube = v; rebuild(); }, 0.5, 0),
      num('blur mm', D.blur ?? ((D.tube ?? 0) * 1.5), v => { D.blur = v; rebuild(); }, 0.5, 0.1),
      num('diffuser T', D.diffuserT ?? 0.7, v => { D.diffuserT = v; rebuild(); }, 0.05, 0, 1)),
    el('p', { class: 'hint' }, 'cone 120 is a bare SMD strip; a shade lip hides the strip from the front; a tube radius wraps the strip in a diffuser (blur shorter than the pitch shows hot spots)'));
}

function patternSection() {
  const cur = scene.pattern ?? { name: PATTERNS[0].name, source: PATTERNS[0].source };
  const names = PATTERNS.map(p => p.name);
  const isPreset = PATTERNS.some(p => p.name === cur.name && p.source === cur.source);
  const sel = select(isPreset ? names : ['custom', ...names], isPreset ? cur.name : 'custom', v => {
    const p = PATTERNS.find(x => x.name === v);
    if (!p) return;
    scene.pattern = { name: p.name, source: p.source };
    applyPattern(); rebuild(); renderPanel();
  });
  const play = button(playing ? '⏸ pause' : '▶ play', () => { playing = !playing; if (!playing) tracer.reset(); renderPanel(); });
  const scrubEl = range('time s', time % 60, 0, 60, 0.05, v => { time = v; if (!playing) tracer.reset(); });
  scrub = { input: scrubEl.querySelector('input')!, out: scrubEl.querySelector('span')! };
  const src = el('textarea', { spellcheck: 'false', class: 'code' }, cur.source);
  const err = el('div', { class: 'err' }, patternErr);
  const apply = button('apply pattern', () => {
    scene.pattern = { name: 'custom', source: src.value };
    applyPattern(); err.textContent = patternErr; rebuild();
    if (!patternErr) renderPanel();
  }, 'primary');
  const exportMap = button(`export LED map (${compiled.ledCount})`, () => {
    const pts = ledMap(compiled).map(p => p.map(v => +v.toFixed(2)));
    const blob = new Blob([JSON.stringify(pts)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `${scene.name.replace(/[^a-z0-9]+/gi, '-')}-ledmap.json`;
    a.click();
  });
  const imgIn = el('input', { type: 'file', accept: 'image/*', hidden: '' });
  imgIn.onchange = async () => {
    const f = imgIn.files?.[0];
    if (!f) return;
    try { mats.setImage(await createImageBitmap(f, { imageOrientation: 'flipY' })); tracer.reset(); status.textContent = ''; }
    catch (e) { status.textContent = `image: ${(e as Error).message}`; }
  };
  return el('section', {},
    el('h2', {}, 'pattern'),
    el('div', { class: 'grid2' }, sel, play),
    range('speed', speed, 0, 4, 0.05, v => { speed = v; }),
    scrubEl,
    el('details', {}, el('summary', {}, 'GLSL'),
      el('pre', { class: 'api' }, PATTERN_API), src, err, apply),
    el('div', { class: 'row' }, exportMap, el('label', { class: 'btn' }, 'load image (uImage)', imgIn)),
  );
}

function renderPanel() { panel.replaceChildren(shellSection(), selectionSection(), patternSection(), actionsSection()); }

// --- picking ---------------------------------------------------------------------------------
let down: { x: number; y: number } | null = null;
function pickAt(x: number, y: number): Sel | null {
  if (job) return null;
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
  if (e.key === ' ') { e.preventDefault(); playing = !playing; if (!playing) tracer.reset(); renderPanel(); }
  if ((e.key === 'Delete' || e.key === 'Backspace') && sel) {
    if (sel.kind === 'cut') scene.cuts.splice(sel.i, 1);
    else if (sel.kind === 'strip') scene.strips.splice(sel.i, 1);
    else if (sel.kind === 'edge') scene.edgeStrips = scene.edgeStrips.filter(x => x.edge !== (sel as Sel).i);
    else if (sel.kind === 'laser') scene.lasers?.splice(sel.i, 1);
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
  scene.eye = { pos: orbit.eye(), target: [...orbit.target] as V3, fov: orbit.fov, exposure: +expIn.value };
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
bounceIn.oninput = () => { tracer.maxBounce = +bounceIn.value; $('bounces-v').textContent = bounceIn.value; tracer.reset(); };
expIn.oninput = () => { tracer.exposure = Math.pow(2, +expIn.value); $('exposure-v').textContent = `${(+expIn.value).toFixed(1)} EV`; };
scaleIn.oninput = () => { renderScale = +scaleIn.value; $('scale-v').textContent = `${renderScale}×`; fit(); };
$<HTMLInputElement>('wire').onchange = e => { wire = (e.target as HTMLInputElement).checked; };
$('render-go').onclick = () => {
  if (job) { finishJob(false); return; }
  const w = Math.max(16, Math.min(8192, Math.round(+$<HTMLInputElement>('render-w').value))),
    h = Math.max(16, Math.min(8192, Math.round(+$<HTMLInputElement>('render-h').value)));
  const target = Math.max(1, Math.round(+$<HTMLInputElement>('render-spp').value));
  const format = $<HTMLSelectElement>('render-fmt').value as 'jpg' | 'png';
  const quality = Math.min(1, Math.max(0.1, +$<HTMLInputElement>('render-q').value / 100));
  if (playing) { playing = false; renderPanel(); }
  job = { w, h, target, format, quality };
  tracer.resize(w, h);
  tracer.reset();
  canvas.style.objectFit = 'contain';
  $('render-go').textContent = 'cancel';
};
/** the scene at the current time as JSON for tools/blender_render.py */
function blenderJSON() {
  const cols = leds.read();
  const data = blenderExport(scene, compiled, orbit.basis(), [...orbit.target] as V3, orbit.fov, +expIn.value,
    i => [cols[i * 4], cols[i * 4 + 1], cols[i * 4 + 2]], m => mats.centre(m));
  return JSON.stringify(data);
}
$('render-blender').onclick = () => {
  const blob = new Blob([blenderJSON()], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `${scene.name.replace(/[^a-z0-9]+/gi, '-')}-blender.json`;
  a.click();
  $('render-status').textContent = 'exported; render with: LC_ALL=C LANG=C blender -b -P tools/blender_render.py -- <file> out.png';
};
function finishJob(save: boolean) {
  if (!job) return;
  if (save) {
    const { w, h, data } = tracer.readback();
    const c = document.createElement('canvas');
    c.width = w; c.height = h;
    c.getContext('2d')!.putImageData(new ImageData(data, w, h), 0, 0);
    const type = job.format === 'png' ? 'image/png' : 'image/jpeg';
    c.toBlob(blob => {
      if (!blob) { status.textContent = 'render: could not encode the image'; return; }
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = `${scene.name.replace(/[^a-z0-9]+/gi, '-')}-${w}x${h}.${job?.format ?? 'jpg'}`;
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 10000);
    }, type, job.quality);
  }
  job = null;
  canvas.style.objectFit = '';
  $('render-go').textContent = 'render to file';
  $('render-status').textContent = save ? 'saved' : 'cancelled';
  fit();
  tracer.reset();
}
orbit.onChange = () => { updateFog(); tracer.reset(); };
bounceIn.oninput(new Event('input')); expIn.oninput(new Event('input'));

new ResizeObserver(fit).observe(canvas);
load(PRESETS[0]());

// --- frame loop: adapt samples per frame to ~16 ms, stop tracing at a sample cap ---------------
const CAP = 8192;
let last = performance.now(), spp = 1, ema = 16;
function tick(now: number) {
  const dt = now - last; last = now;
  const cam = orbit.basis();
  if (playing) time += Math.min(dt, 100) / 1000 * speed;
  leds.run(time);
  mats.run(time);
  tracer.ledTex = leds.tex;
  if (job) {
    // offline render: as many samples as fit in ~40 ms, then save
    const gl = tracer.gl;
    const t0 = performance.now();
    let n = 0;
    while (tracer.frame < job.target && performance.now() - t0 < 40) { tracer.render(cam, 1); n++; }
    gl.flush();
    $('render-status').textContent = `${tracer.frame} / ${job.target} spp · ${job.w}×${job.h}`;
    $('samples').textContent = `rendering ${tracer.frame}/${job.target} spp · ${n}/frame`;
    if (tracer.frame >= job.target) { tracer.render(cam, 0); finishJob(true); }
    requestAnimationFrame(tick);
    return;
  }
  // while animating, the accumulator is a moving average over the last dozen frames
  tracer.keep = playing ? 0.9 : 1;
  const active = playing || tracer.frame < CAP;
  if (active) {
    ema = ema * 0.9 + dt * 0.1;
    if (ema < 12 && spp < 16) spp++; else if (ema > 24 && spp > 1) spp--;
  }
  tracer.render(cam, active ? spp : 0);
  if (playing && scrub) { scrub.input.value = String(time % 60); scrub.out.textContent = (time % 60).toFixed(2); }
  if (wire) overlay.draw(cam, canvas.width, canvas.height);
  $('samples').textContent = `${tracer.frame} spp · ${spp}/frame · ${ema.toFixed(0)} ms · ${canvas.width}×${canvas.height}`;
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);

// debug handles for the console
Object.defineProperties(window, {
  tracer: { value: tracer }, orbit: { value: orbit }, fit: { value: fit }, load: { value: load },
  scene: { get: () => scene }, compiled: { get: () => compiled }, leds: { value: leds }, mats: { value: mats }, blenderJSON: { value: blenderJSON },
  job: { get: () => job }, finishJob: { value: finishJob },
  play: { value: (v: boolean) => { playing = v; renderPanel(); } }, time: { get: () => time, set: (v: number) => { time = v; } },
});
