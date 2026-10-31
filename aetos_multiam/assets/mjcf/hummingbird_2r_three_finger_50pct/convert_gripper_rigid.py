"""Rigid (non-deformable) three-finger gripper at this bundle's own 50% scale.

Ported from the same approach validated against the full-scale standalone
gripper (housing+mount+finger-blade geometry extracted straight from the
flexcomp's own point/tet/surface data, each finger authored as ONE rigid
body -- mount bracket + blade combined -- driven by a PhysicsRevoluteJoint
using the original hinge's own pivot/axis/limits; no deformable schema, no
attachment, works on CPU or GPU dynamics equally).

One thing this bundle needs that the full-scale source didn't: the housing/
mount STL files here are the SAME full-size files (confirmed via sha256 --
identical bytes to the standalone gripper's own housing.stl/finger_mount.stl)
but this bundle's own ``<mesh scale="0.5 0.5 0.5">`` shrinks them at MJCF
load time. The finger *blade* geometry doesn't need this -- flexcomp points
are written directly in the bundle's own already-50%-scale numbers, no STL
asset/scale involved. So STL_SCALE=0.5 is applied only to the housing/mount
points read here, not to the finger blade points.
"""
import math
import struct
from pathlib import Path
import xml.etree.ElementTree as ET
from collections import defaultdict

HERE = Path(__file__).resolve().parent
MJCF_PATH = HERE / "three_finger_gripper_50pct.xml"
GRIPPER_ASSETS = HERE.parent.parent / "meshes" / "mjcf" / HERE.name / "assets" / "gripper"
OUT_DIR = HERE.parent.parent / "meshes" / "usd" / "three_finger_gripper_50pct_rigid"

STL_SCALE = 0.5


def sub(a, b): return [x - y for x, y in zip(a, b)]
def cross(a, b): return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
def dot(a, b): return sum(x*y for x, y in zip(a, b))
def unit(a): return [x/math.sqrt(dot(a, a)) for x in a]
def nums(s): return [float(x) for x in s.split()]
def chunks(a, n): return [a[i:i+n] for i in range(0, len(a), n)]
def det(points, t):
    a, b, c, d = [points[i] for i in t]
    return dot(sub(b, a), cross(sub(c, a), sub(d, a)))
def quat_x_to(axis):
    axis = unit(axis)
    if axis[0] < -0.999999:
        return [0, 0, 1, 0]
    return unit([1+axis[0], 0, -axis[2], axis[1]])
def frame(xy):
    a = nums(xy); x = unit(a[:3]); y0 = a[3:]
    y = unit(sub(y0, [dot(x, y0)*v for v in x])); z = cross(x, y)
    return [[x[i], y[i], z[i]] for i in range(3)]
def rotate(m, p): return [dot(row, p) for row in m]


def stl(path, scale=1.0):
    raw = path.read_bytes(); tris = []
    if len(raw) >= 84 and len(raw) == 84 + 50 * struct.unpack_from('<I', raw, 80)[0]:
        n = struct.unpack_from('<I', raw, 80)[0]
        for i in range(n):
            r = struct.unpack_from('<12fH', raw, 84 + 50*i)
            tris.append(chunks(list(r[3:12]), 3))
    else:
        verts = [nums(line.strip()[7:]) for line in raw.decode().splitlines() if line.strip().startswith('vertex ')]
        tris = chunks(verts, 3)
    points, faces, lookup = [], [], {}
    for tri in tris:
        f = []
        for p in tri:
            p = [v * scale for v in p]
            key = tuple(p)
            if key not in lookup:
                lookup[key] = len(points); points.append(p)
            f.append(lookup[key])
        faces.append(f)
    return {'points': points, 'faces': faces}


def extract():
    r = ET.parse(MJCF_PATH).getroot()
    fingers = []
    body = r.find('./worldbody/body')  # gripper_base
    for b in body.findall('body'):  # finger_1/2/3
        f = b.find('flexcomp')
        p = chunks(nums(f.get('point')), 3)
        ts = chunks([int(x) for x in f.get('element').split()], 4)
        for t in ts:
            if det(p, t) < 0:
                t[0], t[1] = t[1], t[0]
            if det(p, t) <= 1e-18:
                raise ValueError('Degenerate tetrahedron')
        seen = defaultdict(list)
        for t in ts:
            for opposite in range(4):
                face = [v for j, v in enumerate(t) if j != opposite]
                a, c, d = [p[v] for v in face]
                if dot(cross(sub(c, a), sub(d, a)), sub(p[t[opposite]], a)) > 0:
                    face[1], face[2] = face[2], face[1]
                seen[tuple(sorted(face))].append(face)
        surface = [v[0] for v in seen.values() if len(v) == 1]
        j = b.find('joint'); inertia = b.find('inertial'); geom = b.find('geom')
        finger = {
            'name': b.get('name'), 'pivot': nums(b.get('pos')), 'axis': nums(j.get('axis')),
            'limits': nums(j.get('range')), 'points': p, 'tets': ts, 'surface': surface,
            'mass': float(f.get('mass')),
            'mount_mass': float(inertia.get('mass')), 'mount_com': nums(inertia.get('pos')),
            'mount_rotation': frame(geom.get('xyaxes')),
        }
        fingers.append(finger)
    mount = stl(GRIPPER_ASSETS / 'finger_mount.stl', STL_SCALE)
    housing = stl(GRIPPER_ASSETS / 'housing.stl', STL_SCALE)
    # housing_visual geom on gripper_base -- same number the MJCF uses.
    housing_geom = body.find('geom')
    housing_mass = float(housing_geom.get('mass', '0.04375')) if housing_geom is not None else 0.04375
    return {'fingers': fingers, 'housing': housing, 'housing_mass': housing_mass, 'mount': mount}


def val(v):
    if isinstance(v, bool): return str(v).lower()
    if isinstance(v, (int, float)): return format(v, '.12g')
    if isinstance(v, str): return v
    return '(' + ', '.join(val(x) for x in v) + ')'
def arr(v): return '[' + ', '.join(val(x) for x in v) + ']'


class Writer:
    def __init__(self): self.lines = []; self.depth = 0
    def line(self, s): self.lines.append('    '*self.depth + s)
    def begin(self, typ, name, apis=()):
        self.line(f'def {typ} "{name}"' + (' (' if apis else ''))
        if apis:
            self.line('    prepend apiSchemas = ' + arr([f'"{a}"' for a in apis])); self.line(')')
        self.line('{'); self.depth += 1
    def end(self): self.depth -= 1; self.line('}')
    def attr(self, typ, name, value): self.line(f'{typ} {name} = {value}')
    def rel(self, name, path): self.line(f'rel {name} = <{path}>')


def mesh(w, name, points, faces, color):
    w.begin('Mesh', name)
    w.attr('point3f[]', 'points', arr(points))
    w.attr('int[]', 'faceVertexCounts', arr([3]*len(faces)))
    w.attr('int[]', 'faceVertexIndices', arr([x for f in faces for x in f]))
    w.attr('uniform token', 'subdivisionScheme', '"none"')
    w.attr('bool', 'doubleSided', 'true')
    w.attr('color3f[]', 'primvars:displayColor', arr([color]))
    w.end()
def mass(w, m):
    w.attr('float', 'physics:mass', val(m))
def xform(w, pos):
    w.attr('double3', 'xformOp:translate', val(pos))
    w.attr('quatf', 'xformOp:orient', '(1, 0, 0, 0)')
    w.attr('uniform token[]', 'xformOpOrder', arr(['"xformOp:translate"', '"xformOp:orient"']))


def model_rigid(data):
    w = Writer()
    w.line('#usda 1.0'); w.line('(')
    w.line('    defaultPrim = "Gripper"'); w.line('    metersPerUnit = 1'); w.line('    upAxis = "Z"')
    w.line(')')
    w.begin('Xform', 'Gripper')

    # NOT kinematicEnabled: PhysX rejects any kinematic body inside an
    # articulation ("Articulations with kinematic bodies are not
    # supported"), and once mount_child() welds this Housing onto the arm's
    # eef_body_name with a FixedJoint, it becomes part of the SAME merged
    # drone+arm+gripper articulation. A normal dynamic RigidBody held fixed
    # by that weld joint is the correct pattern here (same as every other
    # gripper/arm root body in this project) -- gravity is fine too, it
    # just becomes an internal constraint load once welded, not a fall.
    #
    # Mass MUST be authored: a RigidBody with no PhysicsMassAPI is treated
    # by PhysX as mass=0 / invalid inertia ("small sphere approximated"),
    # which destabilizes the whole merged drone+arm+gripper articulation
    # (the combo tumbles onto its side and falls). Same class of bug as
    # arm_carrier/tool_adapter in import_arm_usd.py.
    w.begin('Xform', 'Housing', ['PhysicsRigidBodyAPI', 'PhysicsMassAPI', 'PhysxRigidBodyAPI'])
    mass(w, data['housing_mass'])
    mesh(w, 'Visual', data['housing']['points'], data['housing']['faces'], (.28, .30, .34))
    w.begin('Mesh', 'Collision', ['PhysicsCollisionAPI', 'PhysicsMeshCollisionAPI'])
    w.attr('point3f[]', 'points', arr(data['housing']['points']))
    w.attr('int[]', 'faceVertexCounts', arr([3]*len(data['housing']['faces'])))
    w.attr('int[]', 'faceVertexIndices', arr([x for f in data['housing']['faces'] for x in f]))
    w.attr('uniform token', 'physics:approximation', '"convexHull"')
    w.end()
    w.end()

    w.begin('Scope', 'Fingers')
    for f in data['fingers']:
        n = f['name']
        mount_pts = [rotate(f['mount_rotation'], p) for p in data['mount']['points']]
        w.begin('Xform', n, ['PhysicsRigidBodyAPI', 'PhysicsMassAPI', 'PhysxRigidBodyAPI'])
        xform(w, f['pivot'])
        mass(w, f['mount_mass'] + f['mass'])
        w.attr('bool', 'physxRigidBody:disableGravity', 'false')
        mesh(w, 'MountVisual', mount_pts, data['mount']['faces'], (.40, .43, .49))
        mesh(w, 'BladeVisual', f['points'], f['surface'], (.56, .59, .64))
        w.begin('Mesh', 'BladeCollision', ['PhysicsCollisionAPI', 'PhysicsMeshCollisionAPI'])
        w.attr('point3f[]', 'points', arr(f['points']))
        w.attr('int[]', 'faceVertexCounts', arr([3]*len(f['surface'])))
        w.attr('int[]', 'faceVertexIndices', arr([x for t in f['surface'] for x in t]))
        w.attr('uniform token', 'physics:approximation', '"convexHull"')
        w.end()
        w.begin('Mesh', 'MountCollision', ['PhysicsCollisionAPI', 'PhysicsMeshCollisionAPI'])
        w.attr('point3f[]', 'points', arr(mount_pts))
        w.attr('int[]', 'faceVertexCounts', arr([3]*len(data['mount']['faces'])))
        w.attr('int[]', 'faceVertexIndices', arr([x for mf in data['mount']['faces'] for x in mf]))
        w.attr('uniform token', 'physics:approximation', '"convexHull"')
        w.end()
        w.end()
    w.end()

    w.begin('Scope', 'Joints')
    for f in data['fingers']:
        w.begin('PhysicsRevoluteJoint', f['name'], ['PhysicsDriveAPI:angular'])
        w.rel('physics:body0', '/Gripper/Housing')
        w.rel('physics:body1', f'/Gripper/Fingers/{f["name"]}')
        w.attr('uniform token', 'physics:axis', '"X"')
        w.attr('point3f', 'physics:localPos0', val(f['pivot']))
        w.attr('point3f', 'physics:localPos1', '(0, 0, 0)')
        q = val(quat_x_to(f['axis']))
        w.attr('quatf', 'physics:localRot0', q)
        w.attr('quatf', 'physics:localRot1', q)
        w.attr('float', 'physics:lowerLimit', '0')
        w.attr('float', 'physics:upperLimit', val(math.degrees(f['limits'][1])))
        w.attr('uniform token', 'drive:angular:physics:type', '"force"')
        for name, v in [('stiffness', 12.0), ('damping', 0.36), ('maxForce', 1.2), ('targetPosition', 0)]:
            w.attr('float', 'drive:angular:physics:' + name, val(v))
        w.end()
    w.end()

    w.end()
    return '\n'.join(w.lines) + '\n'


def main():
    data = extract()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / 'gripper_rigid.usda').write_text(model_rigid(data))
    print('Wrote', OUT_DIR / 'gripper_rigid.usda', flush=True)


if __name__ == '__main__':
    main()
