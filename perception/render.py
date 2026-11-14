"""Blender Cycles renderer for perception scenes: `python -m perception.render job.json` in a Python with bpy.

job.json: {"scenes": [...], "manifest": assets.manifest(...), "karts": karts.json, "geoms": {track: geometry},
"out": dir, "samples": int}. Writes <out>/<scene id>.png (beauty) and .json (truth boxes per kart).
"""
import json
import math
import os
import sys
import time

import bpy
import numpy as np
from mathutils import Matrix

ID_STEP = 0.1   # kart i is drawn with linear red i * ID_STEP in the object-ID pass (Standard view: decoded through the sRGB curve)
LOT_R = 70.0    # static scenes: asphalt out to this radius, grass beyond
PART_SMOOTH_DEG = 35.0


def _mat(name, color=(0.5, 0.5, 0.5), metallic=0.0, rough=0.5, coat=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*color, 1.0)
    b.inputs["Metallic"].default_value = metallic
    b.inputs["Roughness"].default_value = rough
    if "Coat Weight" in b.inputs:
        b.inputs["Coat Weight"].default_value = coat
    return m


def _tex_mat(name, tex, tile_m, tex2=None, tile2=None, lot_r=None, tint=None):
    """PBR ground from Poly Haven maps tiled in world metres; with tex2 and lot_r, tex inside lot_r and tex2 outside; tint recolours."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    coord = nt.nodes.new("ShaderNodeTexCoord")

    def maps(t, tile):
        mp = nt.nodes.new("ShaderNodeMapping")
        mp.inputs["Scale"].default_value = (1 / tile, 1 / tile, 1 / tile)
        nt.links.new(coord.outputs["Object"], mp.inputs["Vector"])
        out = {}
        for k in ("diff", "rough", "nor"):
            n = nt.nodes.new("ShaderNodeTexImage")
            n.image = bpy.data.images.load(t[k], check_existing=True)
            if k != "diff":
                n.image.colorspace_settings.name = "Non-Color"
            nt.links.new(mp.outputs["Vector"], n.inputs["Vector"])
            out[k] = n.outputs["Color"]
        return out

    a = maps(tex, tile_m)
    if tex2 is None:
        col, rough, nor = a["diff"], a["rough"], a["nor"]
    else:
        b = maps(tex2, tile2)
        # radial blend with a noisy edge so the lot boundary is not a perfect circle
        xyz = nt.nodes.new("ShaderNodeSeparateXYZ")
        nt.links.new(coord.outputs["Object"], xyz.inputs["Vector"])
        vl = nt.nodes.new("ShaderNodeVectorMath")
        vl.operation = "LENGTH"
        nt.links.new(coord.outputs["Object"], vl.inputs[0])
        noise = nt.nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = 0.05
        nt.links.new(coord.outputs["Object"], noise.inputs["Vector"])
        add = nt.nodes.new("ShaderNodeMath")
        add.operation = "MULTIPLY_ADD"
        nt.links.new(noise.outputs["Fac"], add.inputs[0])
        add.inputs[1].default_value = 20.0
        nt.links.new(vl.outputs["Value"], add.inputs[2])
        ramp = nt.nodes.new("ShaderNodeMapRange")
        nt.links.new(add.outputs["Value"], ramp.inputs["Value"])
        ramp.inputs["From Min"].default_value = lot_r
        ramp.inputs["From Max"].default_value = lot_r + 3.0

        def mix(x, y, kind):
            n = nt.nodes.new("ShaderNodeMix")
            n.data_type = "RGBA"
            nt.links.new(ramp.outputs["Result"], n.inputs["Factor"])
            nt.links.new(x, n.inputs[6])
            nt.links.new(y, n.inputs[7])
            return n.outputs[2]

        col, rough, nor = mix(a["diff"], b["diff"], 0), mix(a["rough"], b["rough"], 0), mix(a["nor"], b["nor"], 0)
    if tint is not None:   # brightness times a colour: keeps the texture's detail, sets its hue
        bw = nt.nodes.new("ShaderNodeRGBToBW")
        nt.links.new(col, bw.inputs["Color"])
        mul = nt.nodes.new("ShaderNodeMix")
        mul.data_type, mul.blend_type = "RGBA", "MULTIPLY"
        mul.inputs["Factor"].default_value = 1.0
        nt.links.new(bw.outputs["Val"], mul.inputs[6])
        mul.inputs[7].default_value = (*tint, 1.0)
        col = mul.outputs[2]
    nt.links.new(col, bsdf.inputs["Base Color"])
    nt.links.new(rough, bsdf.inputs["Roughness"])
    nm = nt.nodes.new("ShaderNodeNormalMap")
    nt.links.new(nor, nm.inputs["Color"])
    nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    return m


def _kerb_mat():
    m = bpy.data.materials.new("kerb")
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = 0.6
    uv = nt.nodes.new("ShaderNodeUVMap")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(uv.outputs["UV"], sep.inputs["Vector"])
    mul = nt.nodes.new("ShaderNodeMath")
    mul.operation = "MULTIPLY"
    mul.inputs[1].default_value = 0.5
    nt.links.new(sep.outputs["X"], mul.inputs[0])
    fr = nt.nodes.new("ShaderNodeMath")
    fr.operation = "FRACT"
    nt.links.new(mul.outputs["Value"], fr.inputs[0])
    gt = nt.nodes.new("ShaderNodeMath")
    gt.operation = "GREATER_THAN"
    gt.inputs[1].default_value = 0.5
    nt.links.new(fr.outputs["Value"], gt.inputs[0])
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    nt.links.new(gt.outputs["Value"], mix.inputs["Factor"])
    mix.inputs[6].default_value = (0.75, 0.04, 0.03, 1)
    mix.inputs[7].default_value = (0.85, 0.85, 0.85, 1)
    nt.links.new(mix.outputs[2], bsdf.inputs["Base Color"])
    return m


class Renderer:
    def __init__(self, job):
        self.job = job
        self.man = job["manifest"]
        self.karts = job["karts"]
        bpy.ops.wm.read_factory_settings(use_empty=True)
        sc = bpy.context.scene
        self.scene = sc
        sc.render.engine = "CYCLES"
        prefs = bpy.context.preferences.addons["cycles"].preferences
        try:
            prefs.compute_device_type = "METAL" if sys.platform == "darwin" else "CUDA"
            prefs.get_devices()
            for d in prefs.devices:
                d.use = True
            sc.cycles.device = "GPU" if any(d.type != "CPU" for d in prefs.devices) else "CPU"
        except Exception:
            sc.cycles.device = "CPU"
        sc.cycles.samples = job.get("samples", 64)
        sc.cycles.use_adaptive_sampling = True
        sc.cycles.adaptive_threshold = 0.05
        sc.cycles.use_denoising = True
        sc.cycles.max_bounces = 4
        sc.cycles.diffuse_bounces = sc.cycles.glossy_bounces = 2
        sc.cycles.transmission_bounces = sc.cycles.transparent_max_bounces = 4
        sc.render.use_persistent_data = True   # HDRI, textures and BVH of unchanged objects stay on the GPU between frames
        sc.render.resolution_x, sc.render.resolution_y = job["px"]
        sc.render.resolution_percentage = 100
        sc.render.image_settings.file_format = "PNG"
        sc.render.image_settings.color_mode = "RGB"
        sc.view_settings.view_transform = "AgX"
        self.world = bpy.data.worlds.new("world")
        sc.world = self.world
        self.world.use_nodes = True
        self.frame = bpy.data.collections.new("frame")
        sc.collection.children.link(self.frame)
        self.pcol = bpy.data.collections.new("props")   # rebuilt only when the prop list changes, so a sequence keeps its BVH
        sc.collection.children.link(self.pcol)
        self.props_key = None
        self.grounds, self.meshes, self.mats, self.props, self.hdri_sun = {}, {}, {}, {}, {}
        self._materials()
        self.cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
        sc.collection.objects.link(self.cam)
        sc.camera = self.cam

    def _materials(self):
        M = self.mats
        M["chrome"] = _mat("chrome", (0.9, 0.9, 0.9), 1.0, 0.12)
        M["metal"] = _mat("metal", (0.55, 0.55, 0.55), 1.0, 0.4)
        M["alu"] = _mat("alu", (0.8, 0.8, 0.82), 0.9, 0.35)
        M["black"] = _mat("black", (0.02, 0.02, 0.02), 0.0, 0.55)
        M["seat"] = _mat("seat", (0.03, 0.03, 0.03), 0.0, 0.3, 0.6)
        M["motor"] = _mat("motor", (0.2, 0.2, 0.2), 0.6, 0.4)
        M["tire"] = _mat("tire", (0.025, 0.025, 0.025), 0.0, 0.85)
        M["rim"] = _mat("rim", (0.6, 0.6, 0.6), 0.8, 0.45)
        M["cone"] = _mat("cone", (0.9, 0.25, 0.02), 0.0, 0.5)
        M["kerb"] = _kerb_mat()

    def _paint(self, kind, rgb):
        key = (kind, tuple(round(c, 3) for c in rgb))
        if key not in self.mats:
            self.mats[key] = _mat(f"{kind}_{len(self.mats)}", tuple(rgb), 0.3 if kind == "paint" else 0.0, 0.35 if kind == "paint" else 0.3, 0.5)
        return self.mats[key]

    # library meshes

    def _stl(self, name):
        if name not in self.meshes:
            bpy.ops.wm.stl_import(filepath=os.path.join(self.man["cad"], name + ".stl"))
            ob = bpy.context.selected_objects[0]
            me = ob.data
            me.name = name
            bpy.data.objects.remove(ob)
            import bmesh
            bm = bmesh.new()
            bm.from_mesh(me)
            bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
            bm.to_mesh(me)
            bm.free()
            me.shade_smooth()
            me.set_sharp_from_angle(angle=math.radians(PART_SMOOTH_DEG))
            if not me.materials:
                me.materials.append(None)
            self.meshes[name] = me
        return self.meshes[name]

    def _box(self, name, size, bevel=0.02):
        if name not in self.meshes:
            import bmesh
            bm = bmesh.new()
            bmesh.ops.create_cube(bm, size=1.0)
            bmesh.ops.scale(bm, vec=size, verts=bm.verts)
            if bevel:
                bmesh.ops.bevel(bm, geom=bm.edges[:], offset=bevel, segments=3, affect="EDGES")
            me = bpy.data.meshes.new(name)
            bm.to_mesh(me)
            bm.free()
            me.shade_smooth()
            me.materials.append(None)
            self.meshes[name] = me
        return self.meshes[name]

    def _wedge(self, name):
        """Nose cone: a tapered, bevelled wedge in front of the front bumper."""
        if name not in self.meshes:
            import bmesh
            bm = bmesh.new()
            v = [bm.verts.new(p) for p in [(0, -0.36, 0.06), (0, 0.36, 0.06), (0, 0.36, 0.3), (0, -0.36, 0.3),
                                           (0.38, -0.3, 0.07), (0.38, 0.3, 0.07), (0.38, 0.3, 0.16), (0.38, -0.3, 0.16)]]
            for f in [(0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)]:
                bm.faces.new([v[i] for i in f])
            bmesh.ops.bevel(bm, geom=bm.edges[:], offset=0.03, segments=3, affect="EDGES")
            me = bpy.data.meshes.new(name)
            bm.to_mesh(me)
            bm.free()
            me.shade_smooth()
            me.materials.append(None)
            self.meshes[name] = me
        return self.meshes[name]

    def _cone_mesh(self):
        if "cone" not in self.meshes:
            import bmesh
            bm = bmesh.new()
            bmesh.ops.create_cone(bm, cap_ends=True, segments=24, radius1=0.15, radius2=0.025, depth=0.45)
            bmesh.ops.translate(bm, vec=(0, 0, 0.225), verts=bm.verts)
            me = bpy.data.meshes.new("cone")
            bm.to_mesh(me)
            bm.free()
            me.shade_smooth()
            me.materials.append(None)
            self.meshes["cone"] = me
        return self.meshes["cone"]

    def _obj(self, name, mesh, mat, parent, loc=(0, 0, 0), rot=(0, 0, 0)):
        ob = bpy.data.objects.new(name, mesh)
        self.frame.objects.link(ob)
        ob.material_slots[0].link = "OBJECT"
        ob.material_slots[0].material = mat
        ob.parent = parent
        ob.location = loc
        ob.rotation_euler = rot
        return ob

    def kart(self, k, idx):
        """Kart rooted at its rear axle centre on the ground; returns the list of its mesh objects."""
        v = self.karts["variants"][k["variant"]]
        root = bpy.data.objects.new(f"kart{idx}", None)
        self.frame.objects.link(root)
        x, y, yaw = k["pose"]
        root.location = (x, y, 0.0)
        root.rotation_euler = (0, 0, yaw)
        root.scale = (v["scale"],) * 3
        paint, body = self._paint("paint", k["paint"]), self._paint("body", k["body"])
        obs = []
        parts = self.karts["parts"] + (self.karts["tower_parts"] if v["tower"] else [])
        for p in parts:
            mat = paint if p["mat"] == "paint" else self.mats[p["mat"]]
            obs.append(self._obj(p["stl"], self._stl(p["stl"]), mat, root))
        for w in self.karts["wheels"].values():
            wx, wy, wz = w["xyz"]
            for side in (1, -1):
                obs.append(self._obj("tire", self._stl(w["tire"]), self.mats["tire"], root, (wx, side * wy, wz)))
                obs.append(self._obj("rim", self._stl(w["rim"]), self.mats["rim"], root, (wx, side * wy, wz)))
        if v["body"]:
            obs.append(self._obj("nose", self._wedge("nose"), body, root, (1.12, 0, 0)))
            for side in (1, -1):
                obs.append(self._obj("pod", self._box("pod", (0.55, 0.16, 0.2)), body, root, (0.62, side * 0.52, 0.16)))
            obs.append(self._obj("fairing", self._box("fairing", (0.06, 0.4, 0.28), 0.015), body, root, (0.98, 0, 0.3), (0, math.radians(-20), 0)))
            obs.append(self._obj("bumper", self._box("bumper", (0.12, 1.1, 0.12)), body, root, (-0.26, 0, 0.16)))
        if v["mast_m"] > 0:
            m = v["mast_m"]
            obs.append(self._obj("mast", self._box(f"mast{m}", (0.03, 0.03, m), 0.004), self.mats["alu"], root, (0.64, -0.108, 0.681 + m / 2)))
            cyl = self._box("lidar", (0.1, 0.1, 0.08), 0.03)
            obs.append(self._obj("lidar", cyl, self.mats["black"], root, (0.64, -0.108, 0.681 + m - 0.04)))
        return obs

    # props

    def _gltf(self, name):
        if name not in self.props:
            before = set(bpy.data.objects)
            bpy.ops.import_scene.gltf(filepath=self.man["models"][name])
            new = [o for o in bpy.data.objects if o not in before]
            col = bpy.data.collections.new("lib_" + name)
            for o in new:
                for c in list(o.users_collection):
                    c.objects.unlink(o)
                col.objects.link(o)
            pts = np.array([(o.matrix_world @ v.co)[:] for o in new if o.type == "MESH" for v in o.data.vertices])
            lo, hi = pts.min(0), pts.max(0)
            self.props[name] = (col, lo, hi)
        return self.props[name]

    def _instance(self, col, loc, yaw_deg, scale=1.0):
        e = bpy.data.objects.new("inst", None)
        e.instance_type = "COLLECTION"
        e.instance_collection = col
        e.location = loc
        e.rotation_euler = (0, 0, math.radians(yaw_deg))
        e.scale = (scale,) * 3
        self.pcol.objects.link(e)
        return e

    def prop(self, p):
        kind, (x, y, z), yaw = p["kind"], p["xyz"], p["yaw_deg"]
        rng = np.random.default_rng(p["seed"])
        if kind == "cone":
            ob = bpy.data.objects.new("cone", self._cone_mesh())
            self.pcol.objects.link(ob)
            ob.material_slots[0].link = "OBJECT"
            ob.material_slots[0].material = self.mats["cone"]
            ob.location = (x, y, z)
            return
        name = self.man["props"]["tyre" if kind in ("tyre_wall", "tyre_stack") else kind]
        col, lo, hi = self._gltf(name)
        if kind in ("tyre_wall", "tyre_stack"):
            h, dia = hi[2] - lo[2], max(hi[0] - lo[0], hi[1] - lo[1])
            stacks = 1 if kind == "tyre_stack" else int(rng.integers(4, 9))
            c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
            for i in range(stacks):
                off = (i - (stacks - 1) / 2) * dia
                for j in range(int(rng.integers(3, 6)) if kind == "tyre_wall" else 4):
                    self._instance(col, (x + c * off, y + s * off, z + j * h - lo[2]), float(rng.uniform(0, 360)))
            return
        self._instance(col, (x, y, z - lo[2]), yaw)

    # ground, track, world

    def ground(self, env, track, kerbs=True):
        tint = env.get("grass_tint")
        key = (track, env["ground"], env["grass"], tuple(tint or ()), kerbs)
        for k, obs in self.grounds.items():
            for o in obs:
                o.hide_render = k != key
        if key in self.grounds:
            return
        obs = []
        me = bpy.data.meshes.new("ground")
        R = 400.0
        me.from_pydata([(-R, -R, 0), (R, -R, 0), (R, R, 0), (-R, R, 0)], [], [(0, 1, 2, 3)])
        g = bpy.data.objects.new("ground", me)
        bpy.context.scene.collection.objects.link(g)
        tex = self.man["tex"]
        if track is None:
            g.data.materials.append(_tex_mat(f"lot_{env['ground']}", tex[env["ground"]], 4.0, tex[env["grass"]], 3.0, LOT_R))
        else:
            g.data.materials.append(_tex_mat(f"grass_{env['grass']}_{tint}", tex[env["grass"]], 3.0, tint=tint))
            geom = self.job["geoms"][track]
            cu = bpy.data.curves.new("track", "CURVE")
            cu.dimensions = "2D"
            cu.fill_mode = "BOTH"
            for poly in (geom["outer"], geom["inner"]):
                sp = cu.splines.new("POLY")
                sp.points.add(len(poly) - 1)
                for pt, (px, py) in zip(sp.points, poly):
                    pt.co = (px, py, 0.0, 1.0)
                sp.use_cyclic_u = True
            co = bpy.data.objects.new("track_c", cu)
            bpy.context.scene.collection.objects.link(co)
            dg = bpy.context.evaluated_depsgraph_get()
            tm = bpy.data.meshes.new_from_object(co.evaluated_get(dg))
            bpy.data.objects.remove(co)
            t = bpy.data.objects.new("track", tm)
            t.location.z = 0.003
            tm.materials.append(_tex_mat(f"asphalt_{env['ground']}", tex[env["ground"]], 4.0))
            bpy.context.scene.collection.objects.link(t)
            obs.append(t)
            for kb in geom["kerbs"] if kerbs else []:
                pts, nrm, w = np.array(kb["pts"]), np.array(kb["nrm"]), geom["kerb_w"]
                n = len(pts)
                verts = [(*p, 0.0) for p in pts] + [(*(p + w * q), 0.0) for p, q in zip(pts, nrm)]
                faces = [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)]
                km = bpy.data.meshes.new("kerb")
                km.from_pydata(verts, [], faces)
                uv = km.uv_layers.new()
                s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])
                for f, poly in zip(faces, km.polygons):
                    for li, vi in zip(poly.loop_indices, f):
                        uv.data[li].uv = (s[vi % n] if (vi % n) or f[0] == 0 else s[-1] + np.hypot(*(pts[0] - pts[-1])), 0)
                ko = bpy.data.objects.new("kerb", km)
                ko.location.z = 0.006
                km.materials.append(self.mats["kerb"])
                bpy.context.scene.collection.objects.link(ko)
                obs.append(ko)
        obs.append(g)
        self.grounds[key] = obs

    def _sun_azimuth(self, img):
        """World azimuth (rad) of the brightest HDRI texel under Blender's equirect mapping."""
        if img.name not in self.hdri_sun:
            w, h = img.size
            px = np.empty(w * h * 4, np.float32)
            img.pixels.foreach_get(px)
            lum = px.reshape(h, w, 4)[..., :3].sum(-1)
            r, c = np.unravel_index(np.argmax(lum[h // 2:]), lum[h // 2:].shape)   # rows run bottom-up: search the upper half
            u = (c + 0.5) / w
            self.hdri_sun[img.name] = math.pi - 2 * math.pi * (u - 0.5)
        return self.hdri_sun[img.name]

    def world_env(self, env, cam_yaw_deg):
        nt = self.world.node_tree
        nt.nodes.clear()
        img = bpy.data.images.load(self.man["hdri"][env["hdri"]], check_existing=True)
        tc = nt.nodes.new("ShaderNodeTexCoord")
        mp = nt.nodes.new("ShaderNodeMapping")
        et = nt.nodes.new("ShaderNodeTexEnvironment")
        et.image = img
        bg = nt.nodes.new("ShaderNodeBackground")
        out = nt.nodes.new("ShaderNodeOutputWorld")
        nt.links.new(tc.outputs["Generated"], mp.inputs["Vector"])
        nt.links.new(mp.outputs["Vector"], et.inputs["Vector"])
        nt.links.new(et.outputs["Color"], bg.inputs["Color"])
        nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
        # the lookup vector is rotated by rho, so a texel at azimuth a appears at a - rho: put the sun at cam_yaw + rel
        rho = self._sun_azimuth(img) - math.radians(cam_yaw_deg + env["sun_azimuth_deg"])
        mp.inputs["Rotation"].default_value = (0, 0, rho)
        self.scene.view_settings.exposure = env.get("exposure", 0.0)

    def camera(self, c):
        cd = self.cam.data
        cd.sensor_fit = "HORIZONTAL"
        cd.angle = math.radians(c["hfov_deg"])
        cd.clip_start, cd.clip_end = 0.05, 1000.0
        B = Matrix(((0, 0, -1), (-1, 0, 0), (0, 1, 0)))   # camera axes (x right, y up, -z forward) in body axes (x fwd, y left, z up)
        R = (Matrix.Rotation(math.radians(c["yaw_deg"]), 3, "Z") @ Matrix.Rotation(math.radians(c["pitch_down_deg"]), 3, "Y")
             @ Matrix.Rotation(math.radians(c["roll_deg"]), 3, "X") @ B)
        self.cam.matrix_world = Matrix.Translation(c["xyz"]) @ R.to_4x4()

    # passes

    def _id_pass(self, path, show):
        """Workbench flat object-colour render; objects not in show are hidden. Returns (H, W) kart index image."""
        sc = self.scene
        hidden = []
        for o in sc.objects:
            vis = o in show or o.type in ("CAMERA",)
            if not vis and not o.hide_render:
                o.hide_render = True
                hidden.append(o)
        sc.render.engine = "BLENDER_WORKBENCH"
        sh = sc.display.shading
        sh.light, sh.color_type = "FLAT", "OBJECT"
        sc.display.render_aa = "OFF"
        sc.render.film_transparent = True
        vt = sc.view_settings.view_transform
        sc.view_settings.view_transform = "Standard"
        sc.render.image_settings.color_mode = "RGBA"
        exp = sc.view_settings.exposure
        sc.view_settings.exposure = 0.0
        sc.render.filepath = path
        bpy.ops.render.render(write_still=True)
        sc.view_settings.view_transform, sc.view_settings.exposure = vt, exp
        sc.render.image_settings.color_mode = "RGB"
        sc.render.film_transparent = False
        sc.render.engine = "CYCLES"
        for o in hidden:
            o.hide_render = False
        im = bpy.data.images.load(path)
        w, h = im.size
        px = np.empty(w * h * 4, np.float32)
        im.pixels.foreach_get(px)
        bpy.data.images.remove(im)
        os.remove(path)
        px = px.reshape(h, w, 4)[::-1]   # top row first
        lin = np.where(px[..., 0] <= 0.04045, px[..., 0] / 12.92, ((px[..., 0] + 0.055) / 1.055) ** 2.4)   # undo the sRGB encoding of the Standard view
        return np.where(px[..., 3] > 0.5, np.rint(lin / ID_STEP), 0).astype(int)

    def render(self, s):
        t0 = time.time()
        for o in list(self.frame.objects):
            bpy.data.objects.remove(o)
        self.world_env(s["env"], s["camera"]["yaw_deg"])
        self.ground(s["env"], s["track"], s.get("kerbs", True))
        self.camera(s["camera"])
        groups = []
        for i, k in enumerate(s["karts"]):
            if k["role"] == "ego" and not self.karts["ego"].get("render", True):
                continue
            obs = self.kart(k, i)
            for o in obs:
                o.color = (0.0, 0.0, 0.0, 1.0)
            groups.append((k, obs))
        key = json.dumps(s["props"], sort_keys=True)
        if key != self.props_key:
            for o in list(self.pcol.objects):
                bpy.data.objects.remove(o)
            for p in s["props"]:
                self.prop(p)
            self.props_key = key
        out = self.job["out"]
        self.scene.render.filepath = os.path.join(out, s["id"] + ".png")
        self.scene.render.engine = "CYCLES"
        bpy.ops.render.render(write_still=True)
        t1 = time.time()
        for i, (k, obs) in enumerate(groups):
            for o in obs:
                o.color = ((i + 1) * ID_STEP, 0.0, 0.0, 1.0)
        everything = set(self.scene.objects) - {o for o in self.scene.objects if o.hide_render}
        ids = self._id_pass(os.path.join(out, s["id"] + "_id.png"), everything)
        truth = []
        for i, (k, obs) in enumerate(groups):
            vis = ids == i + 1
            truth_i = s["karts"].index(k)
            alone = self._id_pass(os.path.join(out, s["id"] + f"_a{i}.png"), set(obs)) == i + 1 if k["role"] != "ego" else vis
            truth.append({"idx": truth_i, "role": k["role"], "variant": k["variant"], "visible_px": int(vis.sum()), "alone_px": int(alone.sum()),
                          "bbox_visible": _bbox(vis), "bbox_amodal": _bbox(alone)})
        with open(os.path.join(out, s["id"] + ".json"), "w") as f:
            json.dump({"id": s["id"], "px": self.job["px"], "karts": truth, "render_s": round(t1 - t0, 2),
                       "id_s": round(time.time() - t1, 2)}, f)


def _bbox(mask):
    if not mask.any():
        return None
    rows, cols = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
    return [int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1]


def main(path):
    with open(path) as f:
        job = json.load(f)
    os.makedirs(job["out"], exist_ok=True)
    r = Renderer(job)
    todo = [s for s in job["scenes"] if not os.path.exists(os.path.join(job["out"], s["id"] + ".json"))]
    for n, s in enumerate(todo):
        r.render(s)
        print(f"rendered {n + 1}/{len(todo)} {s['id']}", flush=True)


if __name__ == "__main__":
    main(sys.argv[-1])
