"""Render an infinity-designer scene with Blender / Cycles.

    LC_ALL=C LANG=C blender -b -P tools/blender_render.py -- scene-blender.json out.png \
        [--samples 512] [--res 1920 1080] [--exposure -2] [--jpg 92] [--blend scene.blend]

    LC_ALL=C LANG=C blender -P tools/blender_render.py -- scene-blender.json --interactive
        builds the scene in Blender's window with a Cycles rendered viewport looking through
        the app's camera, for orbiting, tweaking and rendering by hand (F12).

The JSON comes from "export for Blender" in the app: the compiled scene at the current
time in millimetres. Polygons carry per-channel reflect / transmit / emit, which map to
Add(Glossy(reflect), Transparent(transmit), Emission(emit)); matte faces are Diffuse; LEDs
are small spheres, diffuser tubes are cylinders coloured per segment, laser lines and beams
are thin emissive cylinders; fog is a Principled Volume filling the shell. The camera and
exposure match the app's view.

Gotchas baked in: force the C locale (an OCIO locale bug segfaults headless Blender),
`refresh_devices()` for GPU on Blender 5, no compositor nodes (moved in Blender 5), and only
NVIDIA backends are probed: Blender 5.2's oneAPI / Level Zero probe segfaults on this
machine, so use /opt/blender-5.0.1-linux-x64/blender if `blender` is 5.2.
"""
import json, math, os, sys
import bpy
from mathutils import Matrix, Vector

MM = 0.001  # scene units are millimetres, Blender works in metres


def args():
    a = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    if not a:
        print(__doc__); sys.exit(1)
    o = {'json': a[0], 'out': None, 'samples': 512, 'res': (1920, 1080), 'exposure': None, 'jpg': None,
         'blend': None, 'interactive': False}
    i = 1
    if i < len(a) and not a[i].startswith('--'): o['out'] = a[i]; i += 1
    while i < len(a):
        k = a[i]
        if k == '--samples': o['samples'] = int(a[i + 1]); i += 2
        elif k == '--res': o['res'] = (int(a[i + 1]), int(a[i + 2])); i += 3
        elif k == '--exposure': o['exposure'] = float(a[i + 1]); i += 2
        elif k == '--jpg': o['jpg'] = int(a[i + 1]); i += 2
        elif k == '--blend': o['blend'] = a[i + 1]; i += 2
        elif k == '--interactive': o['interactive'] = True; i += 1
        else: print('unknown arg', k); sys.exit(1)
    if not o['out'] and not o['blend'] and not o['interactive']:
        print(__doc__); sys.exit(1)
    return o


def v(p):
    return Vector((p[0] * MM, p[1] * MM, p[2] * MM))


def clear():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def gpu():
    prefs = bpy.context.preferences.addons['cycles'].preferences
    for kind in ('OPTIX', 'CUDA'):
        try:
            prefs.compute_device_type = kind
            prefs.refresh_devices()
        except Exception:
            continue
        found = [d for d in prefs.devices if d.type == kind]
        if found:
            for d in prefs.devices: d.use = d.type == kind
            bpy.context.scene.cycles.device = 'GPU'
            print('CYCLES device:', kind, [d.name for d in found])
            return
    print('CYCLES device: CPU')


def node_tree(mat):
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes): nt.nodes.remove(n)
    return nt


def surface_material(name, reflect, transmit, emit, albedo, kind):
    mat = bpy.data.materials.new(name)
    nt = node_tree(mat)
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    if kind == 'matte':
        d = nt.nodes.new('ShaderNodeBsdfDiffuse')
        d.inputs['Color'].default_value = (albedo, albedo, albedo, 1)
        nt.links.new(d.outputs[0], out.inputs['Surface'])
        return mat
    g = nt.nodes.new('ShaderNodeBsdfGlossy'); g.inputs['Roughness'].default_value = 0.0
    g.inputs['Color'].default_value = (*reflect, 1)
    t = nt.nodes.new('ShaderNodeBsdfTransparent'); t.inputs['Color'].default_value = (*transmit, 1)
    add = nt.nodes.new('ShaderNodeAddShader')
    nt.links.new(g.outputs[0], add.inputs[0]); nt.links.new(t.outputs[0], add.inputs[1])
    last = add
    if max(emit) > 0:
        e = nt.nodes.new('ShaderNodeEmission'); e.inputs['Color'].default_value = (*[min(1, c / max(emit)) for c in emit], 1)
        e.inputs['Strength'].default_value = max(emit)
        add2 = nt.nodes.new('ShaderNodeAddShader')
        nt.links.new(last.outputs[0], add2.inputs[0]); nt.links.new(e.outputs[0], add2.inputs[1])
        last = add2
    nt.links.new(last.outputs[0], out.inputs['Surface'])
    return mat


def emitter_material():
    """one emissive material for every LED / tube / beam: colour and strength come from the object"""
    mat = bpy.data.materials.new('emitter')
    nt = node_tree(mat)
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    info = nt.nodes.new('ShaderNodeObjectInfo')
    rad = nt.nodes.new('ShaderNodeAttribute'); rad.attribute_type = 'OBJECT'; rad.attribute_name = 'radiance'
    e = nt.nodes.new('ShaderNodeEmission')
    nt.links.new(info.outputs['Color'], e.inputs['Color'])
    nt.links.new(rad.outputs['Fac'], e.inputs['Strength'])
    nt.links.new(e.outputs[0], out.inputs['Surface'])
    return mat


def body_material():
    mat = bpy.data.materials.new('pcb')
    nt = node_tree(mat)
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    d = nt.nodes.new('ShaderNodeBsdfDiffuse'); d.inputs['Color'].default_value = (0.03, 0.03, 0.03, 1)
    nt.links.new(d.outputs[0], out.inputs['Surface'])
    return mat


def volume_material(fog):
    mat = bpy.data.materials.new('fog')
    nt = node_tree(mat)
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    pv = nt.nodes.new('ShaderNodeVolumePrincipled')
    pv.inputs['Color'].default_value = (*fog['albedo'], 1)
    pv.inputs['Density'].default_value = fog['density']        # per metre, same as the app
    pv.inputs['Anisotropy'].default_value = fog['g']
    pv.inputs['Absorption Color'].default_value = (0, 0, 0, 1)
    nt.links.new(pv.outputs[0], out.inputs['Volume'])
    return mat


def add_mesh(name, verts, faces, mat, coll):
    me = bpy.data.meshes.new(name)
    me.from_pydata([v(p) for p in verts], [], faces)
    me.update()
    ob = bpy.data.objects.new(name, me)
    ob.data.materials.append(mat)
    coll.objects.link(ob)
    return ob


def unit_sphere():
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=1.0)
    ob = bpy.context.active_object
    me = ob.data
    bpy.data.objects.remove(ob)
    return me


def unit_cylinder():
    bpy.ops.mesh.primitive_cylinder_add(vertices=16, radius=1.0, depth=2.0)
    ob = bpy.context.active_object
    me = ob.data
    bpy.data.objects.remove(ob)
    return me


def place_cylinder(name, me, a, b, radius, coll, mat, color=None, radiance=None):
    a, b = v(a), v(b)
    d = b - a
    L = max(d.length, 1e-6)
    ob = bpy.data.objects.new(name, me)
    ob.location = (a + b) / 2
    ob.rotation_mode = 'QUATERNION'
    ob.rotation_quaternion = d.to_track_quat('Z', 'Y')
    ob.scale = (radius * MM, radius * MM, L / 2)
    if color is not None:
        m = max(color) or 1
        ob.color = (color[0] / m, color[1] / m, color[2] / m, 1)
        ob['radiance'] = float(radiance if radiance is not None else m)
    ob.data = me
    if not me.materials: me.materials.append(mat)
    coll.objects.link(ob)
    return ob


def main():
    o = args()
    scene = json.load(open(o['json']))
    clear()
    sc = bpy.context.scene
    coll = bpy.context.collection
    sc.render.engine = 'CYCLES'
    gpu()
    sc.cycles.samples = o['samples']
    sc.cycles.use_denoising = True
    sc.cycles.max_bounces = 64; sc.cycles.glossy_bounces = 64; sc.cycles.transparent_max_bounces = 64
    sc.cycles.transmission_bounces = 64; sc.cycles.volume_bounces = 4
    sc.cycles.caustics_reflective = True; sc.cycles.caustics_refractive = True
    sc.render.resolution_x, sc.render.resolution_y = o['res']
    sc.render.resolution_percentage = 100
    sc.render.film_transparent = False
    sc.view_settings.view_transform = 'AgX'
    sc.view_settings.exposure = o['exposure'] if o['exposure'] is not None else scene.get('exposure', -2)

    # black world
    world = bpy.data.worlds.new('world'); sc.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get('Background')
    if bg: bg.inputs['Color'].default_value = (0, 0, 0, 1); bg.inputs['Strength'].default_value = 0

    # surfaces
    for i, p in enumerate(scene['polys']):
        if p['kind'] == 'portal': continue
        mat = surface_material(f"m{i}", p.get('reflect', [0, 0, 0]), p.get('transmit', [0, 0, 0]), p.get('emit', [0, 0, 0]), p.get('albedo', 0.04), p['kind'])
        add_mesh(f"poly{i}", p['ring'], [list(range(len(p['ring'])))], mat, coll)

    # fog: the shell as a volume container
    if scene.get('fog') and scene.get('shell'):
        sh = scene['shell']
        add_mesh('fog', sh['verts'], sh['faces'], volume_material(scene['fog']), coll)

    # emitters
    em = emitter_material(); pcb = body_material()
    sph = unit_sphere(); cyl = unit_cylinder()
    sph.materials.append(em)
    cyl_e = cyl.copy(); cyl_e.materials.append(em)
    cyl_b = cyl.copy(); cyl_b.materials.append(pcb)
    for i, e in enumerate(scene['emitters']):
        if e['type'] == 'led':
            ob = bpy.data.objects.new(f"led{i}", sph)
            ob.location = v(e['pos']); ob.scale = (e['radius'] * MM,) * 3
            c = e['color']; m = max(c) or 1
            ob.color = (c[0] / m, c[1] / m, c[2] / m, 1); ob['radiance'] = float(m)
            coll.objects.link(ob)
        elif e['type'] == 'tube':
            place_cylinder(f"tube{i}", cyl_e, e['a'], e['b'], e['radius'], coll, em, e['color'])
        elif e['type'] == 'body':
            place_cylinder(f"body{i}", cyl_b, e['a'], e['b'], e['radius'], coll, pcb)
    for i, b in enumerate(scene.get('beams', [])):
        place_cylinder(f"beam{i}", cyl_e, b['a'], b['b'], b['radius'], coll, em, b['color'], b['radiance'])

    # camera
    cam = scene['camera']
    cd = bpy.data.cameras.new('cam'); cd.sensor_fit = 'VERTICAL'; cd.angle_y = math.radians(cam['fov'])
    cd.clip_start = 0.001; cd.clip_end = 100
    co = bpy.data.objects.new('cam', cd); coll.objects.link(co); sc.camera = co
    # the scene is Y-up, so build the look-at frame by hand (to_track_quat assumes Z-up)
    pos = v(cam['pos'])
    fwd = (v(cam['target']) - pos).normalized()
    right = fwd.cross(Vector((0, 1, 0)))
    if right.length < 1e-6: right = Vector((1, 0, 0))
    right.normalize()
    up = right.cross(fwd)
    m = Matrix((right, up, -fwd)).transposed().to_4x4()
    m.translation = pos
    co.matrix_world = m

    sc.cycles.preview_samples = 64
    sc.cycles.use_preview_denoising = True
    sc.render.image_settings.file_format = 'JPEG' if o['jpg'] else 'PNG'
    if o['jpg']: sc.render.image_settings.quality = o['jpg']
    if o['out']: sc.render.filepath = os.path.abspath(o['out'])

    if o['blend']:
        path = os.path.abspath(o['blend'])
        bpy.ops.wm.save_as_mainfile(filepath=path)
        print('saved', path)
    if o['out']:
        bpy.ops.render.render(write_still=True)
        print('wrote', sc.render.filepath)
    if o['interactive'] and not bpy.app.background:
        bpy.app.timers.register(lambda: viewport(co), first_interval=0.2)


def viewport(cam_ob):
    """once the window exists: rendered shading, looking through the scene camera"""
    wm = bpy.context.window_manager
    for win in wm.windows:
        for area in win.screen.areas:
            if area.type != 'VIEW_3D': continue
            for space in area.spaces:
                if space.type != 'VIEW_3D': continue
                space.shading.type = 'RENDERED'
                space.region_3d.view_perspective = 'CAMERA'
                space.clip_start = 0.001
                space.lock_camera = True     # orbiting moves the camera itself, so F12 renders what you see
    bpy.context.scene.camera = cam_ob
    return None


main()
