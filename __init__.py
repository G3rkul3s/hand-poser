bl_info = {
    "name": "Hands Pose and Render",
    "author": "Nikita Morev",
    "version": (0, 1),
    "blender": (2, 80, 0),
    "location": "3D Viewport > Sidebar (N-Panel) > Hands Poser",
    "description": "", # TODO: add description
    "warning": "",
    "doc_url": "",
    "category": "Hands Poser",
}

import bpy
import random
import numpy as np
import math
import json
# import re
# import os
# import sys
# import typing
from math import radians
from mathutils import Vector, Quaternion, Matrix

from pathlib import Path
ROOT_DIR = Path(__file__).parent

from importlib import reload
from . import load_mano as lm

_cached_poses = [("NONE", "None", "")]
_cached_shapes = [("NONE", "None", "")]
_cached_pose_attachments = [("NONE", "None", "")]

SMPLX_JOINT_NAMES = [
    'pelvis','left_hip','right_hip','spine1','left_knee','right_knee','spine2','left_ankle','right_ankle','spine3', 'left_foot','right_foot','neck','left_collar','right_collar','head','left_shoulder','right_shoulder','left_elbow', 'right_elbow','left_wrist','right_wrist',
    'jaw','left_eye_smplhf','right_eye_smplhf','left_index1','left_index2','left_index3','left_middle1','left_middle2','left_middle3','left_pinky1','left_pinky2','left_pinky3','left_ring1','left_ring2','left_ring3','left_thumb1','left_thumb2','left_thumb3','right_index1','right_index2','right_index3','right_middle1','right_middle2','right_middle3','right_pinky1','right_pinky2','right_pinky3','right_ring1','right_ring2','right_ring3','right_thumb1','right_thumb2','right_thumb3'
]
NUM_SMPLX_JOINTS=len(SMPLX_JOINT_NAMES)
NUM_MANO_JOINTS=len(lm.BONE_NAMES)

def reload_modules():
    # print("reloading")
    reload(lm)

# TODO: delete all print statements
# TODO: delete me later ???
"""def ensure_site_packages(packages: typing.List[typing.Tuple[str, str]]):    
    if not packages:
        return

    import site
    import importlib
    import importlib.util

    user_site_packages = site.getusersitepackages()
    sys.path.append(user_site_packages)

    modules_to_install = [module[1] for module in packages if not importlib.util.find_spec(module[0])]
    if not modules_to_install:
        return

    if bpy.app.version < (2,91,0):
        python_binary = bpy.app.binary_path_python
    else:
        python_binary = sys.executable
        
    import subprocess
    subprocess.run([python_binary, '-m', 'ensurepip'], check=True)
    subprocess.run([python_binary, '-m', 'pip', 'install', *modules_to_install, "--user"], check=True)
    
    importlib.invalidate_caches()

ensure_site_packages([
    # ("flatbuffers", "flatbuffers"),
    ("torch", "torch"),
])"""

# from manotorch.manolayer import ManoLayer, MANOOutput
# from .VIRTOSHA.FlatBuffers import FrameBatch

def compositing_error(self, context):
    self.layout.label(text="Compositing is not configured")

def sensor_collection_error(self, context):
    self.layout.label(text='"Sensors" collection not found')

def update_light_selection(self, context):
    """
    This function is called whenever 'light_selection' changes.
    """
    sensor_collection = bpy.data.collections.get('Sensors')
    if not sensor_collection:
        bpy.context.window_manager.popup_menu(sensor_collection_error, title="Warning", icon='ERROR')
        return
    sensor_objects = set(sensor_collection.objects)
    selection = self.light_selection
    nl_render = True if selection == 'RGB' else False
    for obj in sensor_objects:
        if obj.type == 'LIGHT':
            obj.hide_render = nl_render
            # if self.viewport_checkbox:
            obj.hide_viewport = nl_render
    for obj in bpy.data.objects:
        if obj.type == 'LIGHT' and obj not in sensor_objects:
            obj.hide_render = not nl_render
            # if self.viewport_checkbox:
            obj.hide_viewport = not nl_render
    tree = context.scene.node_tree
    hue_correct = tree.nodes.get("BlackAndWhiteFilter")
    mix_rgb = tree.nodes.get("DepthImage")
    composite = tree.nodes.get("Composite")
    if selection == 'DEPTH' and composite and mix_rgb:
        context.view_layer.use_pass_mist = True
        tree.links.new(mix_rgb.outputs['Image'], composite.inputs['Image'])
    elif hue_correct and composite:
        context.view_layer.use_pass_mist = False
        hue_correct.mute = nl_render
        tree.links.new(hue_correct.outputs['Image'], composite.inputs['Image'])
    else:
        bpy.context.window_manager.popup_menu(compositing_error, title="Warning", icon='ERROR')

def update_joint_positions(armature_obj, J_regressor, vert_shaped, context, hand):
    """
    Updates the joint (bone) positions in the armature based on the current shape of the mesh.
    """
    # Compute new joint positions
    joints = J_regressor @ vert_shaped # shape (16, 3)
    active_curr = context.view_layer.objects.active
    context.view_layer.objects.active = armature_obj
    current_mode = context.object.mode
    bpy.ops.object.mode_set(mode='EDIT')
    edit_bones = armature_obj.data.edit_bones

    for i, name in enumerate(lm.BONE_NAMES):
        name = hand + name
        if name not in edit_bones:
            continue
        bone = edit_bones[name]
        new_head = joints[i]
        new_tail = bone.tail + Vector(new_head - bone.head)
        bone.head = new_head
        bone.tail = new_tail
    
    for name, index in lm.FINGERTIPS.items():
        name = hand + name
        if name not in edit_bones:
            continue
        bone = edit_bones[name]
        new_head = vert_shaped[index]
        new_tail = bone.tail + Vector(new_head - bone.head)
        bone.head = new_head
        bone.tail = new_tail

    bpy.ops.object.mode_set(mode=current_mode)
    context.view_layer.objects.active = active_curr

def load_poses_from_file(self, context):
    global _cached_poses
    pose_path = Path(bpy.path.abspath(context.scene.pose_path))
    if pose_path.is_file():
        with open(pose_path, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                _cached_poses = [("NONE", "None", "")]
                return
            poses = []
            poses.extend([(pose["name"], pose["name"], "") for pose in data 
                                if pose["arm_name"] == context.scene.armature_ref.name 
                                and pose["bone_col"] == context.scene.selected_bone_collection])
            _cached_poses = poses if len(poses) > 0 else [("NONE", "None", "")]
            _cached_poses.sort()
    else:
        _cached_poses = [("NONE", "None", "")]
        self.report({'ERROR'}, "A file with saved poses is missing.\nSave a pose to create one")
        return
    return

def load_vis_att_from_file(self, context):
    global _cached_pose_attachments
    if context.scene.pose_selection == 'NONE':
        _cached_pose_attachments = [("NONE", "None", "")]
        return
    data = []
    pose_path = Path(bpy.path.abspath(context.scene.pose_path))
    if pose_path.is_file():
            with open(pose_path, 'r') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    _cached_pose_attachments = [("NONE", "None", "")]
                    return
    else:
        # self.report({'ERROR'}, "A file with saved poses is missing")
        _cached_pose_attachments = [("NONE", "None", "")]
        return
    for pose in data:
            if pose['name'] == context.scene.pose_selection:
                objects = pose.get('render_with')
                if objects is None or len(objects) == 0:
                    _cached_pose_attachments = [("NONE", "None", "")]
                else:
                    _cached_pose_attachments = []
                    _cached_pose_attachments.extend([(obj, obj, "") for obj in objects])
                break

def load_shapes_from_file(self, context):
    global _cached_shapes
    shape_path = Path(bpy.path.abspath(context.scene.shape_path))
    if shape_path.is_file():
        with open(shape_path, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                _cached_shapes = [("NONE", "None", "")]
                return
            shapes = []
            shapes.extend([(shape["name"], shape["name"], "") for shape in data])
            _cached_shapes = shapes if len(shapes) > 0 else [("NONE", "None", "")]
            _cached_shapes.sort()
    else:
        _cached_shapes = [("NONE", "None", "")]
        self.report({'ERROR'}, "A file with saved shapes is missing.\nSave a shape to create one")
        return
    return

def get_visibility_attachments(self, context):
    global _cached_pose_attachments
    return _cached_pose_attachments

def get_poses(self, context):
    global _cached_poses
    return _cached_poses

def get_shapes(self, context):
    global _cached_shapes
    return _cached_shapes

def is_hide_render_keyframed(obj, frame=None):
    if frame is None:
        frame = bpy.context.scene.frame_current
    if not obj.animation_data or not obj.animation_data.action:
        return False

    action = obj.animation_data.action
    # Find the FCurve for hide_render
    fcurve = None
    for fc in action.fcurves:
        if fc.data_path == "hide_render":
            fcurve = fc
            break
    if fcurve is None:
        return False
    # Check if this frame has a keyframe point
    for kp in fcurve.keyframe_points:
        if int(kp.co.x) == frame:
            return True

    return False

"""bpy.types.Scene.viewport_checkbox = bpy.props.BoolProperty(
    name="Update in Viewport",
    description="Display render type in viewport",
    default=True,
    update=update_light_selection
)"""

"""bpy.types.Scene.export_checkbox = bpy.props.BoolProperty(
    name="Export Metadata",
    description="Export sensors and joints positions alongside renders",
    default=True,
#    update=
)"""

bpy.types.Scene.light_selection = bpy.props.EnumProperty(
    name="",
    description="Render type",
    items=[
        ('RGB', "Color (RGB)", "Render with scene lighting"),
        ('IR', "Infrared", "Render with infrared sensor lighting"),
        ('DEPTH', "Depth", "Render depth pass"),
    ],
    default='RGB',
    update=update_light_selection
)

bpy.types.Scene.hand_selection = bpy.props.EnumProperty(
    name="",
    description="MANO hand",
    items=[
        ('LEFT', "Left", "Left hand"),
        ('RIGHT', "Right", "Right hand")
    ],
    default='RIGHT',
)

bpy.types.Scene.file_extension_selection = bpy.props.EnumProperty(
    name="",
    description="File extension",
    items=[
        ('JSON', ".json", "Export to .json"),
        # ('FBS', ".fbs", "Export to .fbs")
    ],
    default='JSON',
)

bpy.types.Scene.pose_selection = bpy.props.EnumProperty(
    name="",
    description="Predefined pose",
    items=get_poses,
    default=0,
    # default=1,
    update=load_vis_att_from_file,
)

bpy.types.Scene.pose_name = bpy.props.StringProperty(
    name="",
    description="Give a name to the pose",
)

bpy.types.Scene.pose_path = bpy.props.StringProperty(
    name="",
    description="",
    subtype='FILE_PATH',
)

bpy.types.Scene.shape_name = bpy.props.StringProperty(
    name="",
    description="Give a name to the shape",
)

bpy.types.Scene.shape_path = bpy.props.StringProperty(
    name="",
    description="",
    subtype='FILE_PATH',
)

bpy.types.Scene.shape_selection = bpy.props.EnumProperty(
    name="",
    description="Predefined shape",
    items=get_shapes,
    default=0,
    # default=1,
)

bpy.types.Scene.pose_attachment = bpy.props.PointerProperty(
    name="",
    description = "An object that should be rendered with the current pose",
    type=bpy.types.Object,
)

bpy.types.Scene.origin_ref = bpy.props.PointerProperty(
    name="",
    description = "Select a reference frame for the sensors and joints positions. \nLeave empty to use world coordinates",
    type=bpy.types.Object,
)

bpy.types.Scene.save_folder = bpy.props.StringProperty(
    name="",
    description="Output path for rendered images and metadata",
    subtype='DIR_PATH'
)

bpy.types.Scene.mano_folder = bpy.props.StringProperty(
    name="",
    description="Path to MANO .npz models.\nUse provided 'unpack_MANO.py' script to generate those files",
    subtype='DIR_PATH'
)

bpy.types.Scene.override_compositing = bpy.props.BoolProperty(
    name="Override",
    description="Override the existing compositing node tree",
    default=False,
#    update=
)

"""bpy.types.Scene.random_poses_slider = bpy.props.IntProperty(
    name="Number of Poses",
    description="Number of random poses to generate and render",
    default=1,
    min=1,
    max=100
)"""

bpy.types.Scene.std_slider = bpy.props.FloatProperty(
    name="Standard Deviation",
    description="How likely it is that a random shape differs from the default one",
    default=1.5,
    min=0.0,
    max=5.0
)

bpy.types.Scene.armature_ref = bpy.props.PointerProperty(
    name="",
    type=bpy.types.Object,
    poll=lambda self, obj: obj.type == 'ARMATURE',
)

bpy.types.Scene.use_poseshapes = bpy.props.BoolProperty(
    name="Corrective Pose Shapes",
    description="Enable/disable corrective pose shapes",
    default=True,
)

bpy.types.Scene.deformable_mesh_right_ref = bpy.props.PointerProperty(
    name="",
    description = "Mesh with MANO right hand vertex group and shape keys",
    type=bpy.types.Object,
    poll=lambda self, obj: obj.type == 'MESH' and obj.data.shape_keys and obj.vertex_groups["MANO_RIGHT_HAND"],
)

bpy.types.Scene.deformable_mesh_left_ref = bpy.props.PointerProperty(
    name="",
    description = "Mesh with MANO left hand vertex group and shape keys",
    type=bpy.types.Object,
    poll=lambda self, obj: obj.type == 'MESH' and obj.data.shape_keys and obj.vertex_groups["MANO_LEFT_HAND"],
)

bpy.types.Scene.sensor_type = bpy.props.EnumProperty(
    name="",
    description="Sensor type",
    items=[
        ('LEAPSTEREO', "Ultraleap Stereo IR 170", "Ultraleap infrared stereo camera"),
        ('FEMTOBOLT', "Orbbec Femto Bolt", "ORBBEC color/depth Time-of-Flight camera"),
        ('KINECT', "MicrosoftKinect v2", "Microsoft color/depth Time-of-Flight camera"),
    ],
    default='LEAPSTEREO',
)

bpy.types.Scene.random_positions_ref = bpy.props.PointerProperty(
    name="",
    description = "A mesh to randomly sample it's vertecies for the sensor placement",
    type=bpy.types.Object,
    poll=lambda self, obj: obj.type == 'MESH',
)

def pose_get_bone_collections(self, context):
    items = [("ALL", "All", "Use the whole armature")]
    armature = self.armature_ref
    if armature and armature.type == 'ARMATURE':
        items.extend([(bc.name, bc.name, f"Use {bc.name} collection") for bc in armature.data.collections])
    return items

bpy.types.Scene.selected_bone_collection = bpy.props.EnumProperty(
    name="",
    description="Bone collection",
    items=pose_get_bone_collections,
    update=load_poses_from_file,
)

bpy.types.Scene.sensor_orientation = bpy.props.EnumProperty(
    name="",
    description="Sensor orientation",
    items=[
        ('KEEP', "Keep", "Keep the original orientation"),
        ('NORMAL', "Normals", "Pointing along the normals of the sample mesh"),
        ('NEGNORMAL', "Negative normals", "Pointing along the negative of the normals of the sample mesh"),
        ('ORIGIN', "Mesh origin", "Pointing to the origin of the sample mesh"),
        ('CURSOR', "3D Cursor", "Pointing to the 3D cursor")
    ],
    default='KEEP',
    # update=,
)

"""bpy.types.Scene.angle_restriction = bpy.props.EnumProperty(
    name="",
    description="Amount of allowed sensor rotation configurations",
    items = [(str(i), str(i), "") for i in range(1, 361) if 360 % i == 0],
    default = "4",
)"""

bpy.types.Scene.keyframe_attachments = bpy.props.BoolProperty(
    name="Keyframe attachments' visibility",
    description="Keyframe visibility of the current attachments",
    default=False,
)

bpy.types.Scene.list_attachments = bpy.props.EnumProperty(
    name="",
    description="Objects that will render with the selected pose after keyframing",
    items=get_visibility_attachments,
)

class VIEW3D_OT_MultiviewRender(bpy.types.Operator):
    """"""
    bl_idname = "view3d.multiview_render"
    bl_label = "Render Animation"
    bl_description="Render the imgaes from sensors"
    
    def execute(self, context):
        # Check for sensors
        sensor_collection = bpy.data.collections.get('Sensors')
        if not sensor_collection:
            self.report({'ERROR'}, "Sensors collection not found")
            return {'CANCELLED'}
        # Check for save folder
        save_folder = context.scene.save_folder
        if not save_folder:
            self.report({'ERROR'}, "Save folder not selected")
            return {'CANCELLED'}
        # Create folder where rendered images are saved if it doesn't exists
        render_type = context.scene.light_selection
        if render_type == 'RGB':
            rgb_path = Path(bpy.path.abspath(save_folder), "rgb")
            rgb_path.mkdir(exist_ok=True)
        elif render_type == 'IR':
            ir_path = Path(bpy.path.abspath(save_folder), "infrared")
            ir_path.mkdir(exist_ok=True)
        elif render_type == 'DEPTH':
            depth_path = Path(bpy.path.abspath(save_folder), "depth")
            depth_path.mkdir(exist_ok=True)
        '''
        # Set up multiview render
        # NOTE: disabled for now
        context.scene.render.use_multiview = True
        sensor_names = {obj.name for obj in sensor_collection.objects if obj.type == 'CAMERA'}
        views = context.scene.render.views
        for v in list(views):
            if v.name == 'left' or v.name == 'right':
                v.use = False
            elif v.name not in sensor_names:
            #    views.remove(v)
                v.use = False
            # else:
            #     v.use = True
        '''
        cameras = []
        lights = []
        try:
            if render_type == 'RGB':
                cameras = [obj for obj in sensor_collection.objects if obj.type == 'CAMERA' and obj["in use"] and obj["color"]]
            elif render_type == 'IR':
                cameras = [obj for obj in sensor_collection.objects if obj.type == 'CAMERA' and obj["in use"] and obj["infrared"]]
                lights = [obj for obj in sensor_collection.objects if obj.type == 'LIGHT']
            elif render_type == 'DEPTH':
                cameras = [obj for obj in sensor_collection.objects if obj.type == 'CAMERA' and obj["in use"] and obj["depth"]]
        except KeyError as e:
            self.report({'ERROR'}, 
                        f"{repr(e)}\nA camera is missing one or all of the custom properties {{'in use', 'color', 'infrared', 'depth'}}")
            return{'CANCELLED'}
        if len(cameras) == 0:
            self.report({'WARNING'}, f"No suitable camera was found for rendering in '{render_type}' mode")
            return {'CANCELLED'}
        # Iterate over all frames
        current_frame = context.scene.frame_current
        for i in range(context.scene.frame_start, context.scene.frame_end+1):
            context.scene.frame_set(i)
            # Iterate over all cameras
            for camera in cameras:
                # Turn on the ir-lights only for the current sensor
                if render_type == 'IR':
                    sensor = camera.parent
                    for light in lights:
                        if light in sensor.children:
                            light.hide_render = False
                        else:
                            light.hide_render = True
                context.scene.camera = camera
                if render_type == 'RGB':
                    context.scene.render.filepath = str(Path(rgb_path, f"Frame_{i:06}_" + camera.name))
                elif render_type == 'IR':
                    context.scene.render.filepath = str(Path(ir_path, f"Frame_{i:06}_" + camera.name))
                elif render_type == 'DEPTH':
                    context.scene.render.filepath = str(Path(depth_path, f"Frame_{i:06}_" + camera.name))
                # Invoke render
                # bpy.ops.render.render(write_still=True)
        for light in lights:
            light.hide_render = False
        context.scene.frame_set(current_frame)

        # context.scene.render.filepath = bpy.path.abspath(folder + "Frame_")
        # bpy.ops.render.render(animation=True)
        self.report({'INFO'}, "Render successfully saved")
        return {'FINISHED'}

class VIEW3D_OT_AddSensor(bpy.types.Operator):
    """"""
    bl_idname = "view3d.add_sensor"
    bl_label = "Add"
    bl_description="Add a sensor to the scene"
    bl_options = {'REGISTER', 'UNDO'}
    
    def add_camera_properties(self, camera, rgb=True, ir=False, depth=True):
        camera["in use"] = True
        camera.id_properties_ui("in use").update(
            description="Should this camera be used for rendering",
            default=True,
        )
        camera["color"] = rgb
        camera.id_properties_ui("color").update(
            description="Does this camera support rendering in RGB color mode",
            default=True,
        )
        camera["infrared"] = ir
        camera.id_properties_ui("infrared").update(
            description="Does this camera support rendering in infrared mode",
            default=False,
        )
        camera["depth"] = depth
        camera.id_properties_ui("depth").update(
            description="Does this camera support rendering in depth mode",
            default=True,
        )
    
    def new_leap_camera(self, cam_name="Camera"):
        cam_data = bpy.data.cameras.new(name=cam_name)
        """
        # == version 1 ==
        cam_data.type = 'PANO'
        cam_data.panorama_type = 'FISHEYE_EQUISOLID'
        cam_data.fisheye_lens = 1.50
        cam_data.fisheye_fov = radians(170.0)
        # == version 2 ==
        cam_data.type = 'PERSP'
        cam_data.lens = 2.1
        cam_data.sensor_fit = 'AUTO'
        # cam_data.sensor_width = 4
        cam_data.sensor_width = 14
        """
        # == version 3 ==
        cam_data.type = 'PERSP'
        cam_data.lens = 6.0
        cam_data.sensor_width = 36
        cam_data.sensor_fit = 'AUTO'

        cam_data.clip_start = 0.01
        cam_data.clip_end = 50
        cam_data.display_size = 0.3

        return cam_data

    def new_femtobolt_camera(self, cam_name="Camera"):
        cam_data = bpy.data.cameras.new(name=cam_name)
        cam_data.type = 'PERSP'
        cam_data.angle = radians(80.0)
        cam_data.sensor_width = 36
        cam_data.sensor_fit = 'AUTO'

        cam_data.clip_start = 0.01
        cam_data.clip_end = 50
        cam_data.display_size = 0.3

        return cam_data

    def new_kinect_camera(self, cam_name="Camera"):
        pass

    def new_ir_light(self, light_name='Spot'):
        spot_data = bpy.data.lights.new(name=light_name, type='SPOT')
        spot_data.energy = 0.5 # TODO: not sure what level should be apropriate
        spot_data.spot_size = radians(180.0)
        spot_data.spot_blend = 0.3
        return spot_data

    def execute(self, context):
        # current_mode = context.mode
        # bpy.ops.object.mode_set(mode='OBJECT')
        collection_name  = "Sensors"
        # Create Sensors collection if it doesn't exist
        target_collection = bpy.data.collections.get(collection_name)
        if not target_collection:
            target_collection = bpy.data.collections.new(collection_name)
            context.scene.collection.children.link(target_collection)
        
        if context.scene.sensor_type == 'LEAPSTEREO':
            base_name = "LeapStereoIR"
            index = 1
            while f"{base_name}.{index:03}" in bpy.data.objects:
                index += 1
            empty_name = f"{base_name}.{index:03}"
            
            # Create an empty
            empty = bpy.data.objects.new(name=empty_name, object_data=None)
            empty.empty_display_type = 'PLAIN_AXES'
            target_collection.objects.link(empty)
            empty.empty_display_size = 0.2

            # Import the camera model
            bpy.ops.wm.obj_import(filepath= str(ROOT_DIR / 'data/UltraleapStereoIR170Casing.obj'), check_existing=True)
            imported_object = bpy.context.selected_objects[0]
            imported_object.parent = empty
            imported_object.name = f"{base_name}Casing"

            # Create cameras
            cam_left_data = self.new_leap_camera()
            cam_left_obj = bpy.data.objects.new(name=f"{base_name}_{index:03}_Camera_Left", 
                                                object_data=cam_left_data)
            cam_left_obj.parent = empty
            cam_left_obj.location = (-0.032, 0.0, 0.0)
            cam_left_obj.scale = (0.2, 0.2, 0.2)
            self.add_camera_properties(cam_left_obj, rgb=False, ir=True)
            target_collection.objects.link(cam_left_obj)
            
            # render = context.scene.render
            # view_names = {v.name for v in render.views}
            # cam_left_name = f"Camera_{base_name}_{index:03}_Left"
            # if cam_left_name not in view_names:
            #     cam_left_rv = render.views.new(name=cam_left_name)
            #     cam_left_rv.camera_suffix = f"_{base_name}_{index:03}_Left"
            #     cam_left_rv.use = True
            #     # cam_left_rv.file_suffix = ""
            
            cam_right_data = self.new_leap_camera()
            cam_right_obj = bpy.data.objects.new(name=f"{base_name}_{index:03}_Camera_Right", 
                                                object_data=cam_right_data)
            cam_right_obj.parent = empty
            cam_right_obj.location = (0.032, 0.0, 0.0)
            cam_right_obj.scale = (0.2, 0.2, 0.2)
            self.add_camera_properties(cam_right_obj, rgb=False, ir=True)
            target_collection.objects.link(cam_right_obj)
            
            # cam_right_name = f"Camera_{base_name}_{index:03}_Right"
            # if cam_right_name not in view_names:
            #     cam_right_rv = render.views.new(name=cam_right_name)
            #     cam_right_rv.camera_suffix = f"_{base_name}_{index:03}_Right"
            #     cam_right_rv.use = True
            #     # cam_right_rv.file_suffix = ""
            
            # Create IR LEDs
            spot_left_data = self.new_ir_light()
            
            spot_obj_left = bpy.data.objects.new(name=f"IR_LED_{index:03}_Left", object_data=spot_left_data)
            spot_obj_left.parent = empty
            spot_obj_left.location = (-0.05, 0.0, 0.0)
            spot_obj_left.scale = (0.01, 0.01, 0.01)
            target_collection.objects.link(spot_obj_left)
            
            spot_right_data = self.new_ir_light()
            
            spot_obj_right = bpy.data.objects.new(name=f"IR_LED_{index:03}_Right", object_data=spot_right_data)
            spot_obj_right.parent = empty
            spot_obj_right.location = (0.05, 0.0, 0.0)
            spot_obj_right.scale = (0.01, 0.01, 0.01)
            target_collection.objects.link(spot_obj_right)

            # Set visibility
            nl_render = True if context.scene.light_selection == 'RGB' else False
            spot_obj_left.hide_render = nl_render
            spot_obj_right.hide_render = nl_render
            # if context.scene.viewport_checkbox:
            spot_obj_left.hide_viewport = nl_render
            spot_obj_right.hide_viewport = nl_render
            
            sensors_view_layer = bpy.context.view_layer.layer_collection.children.get(collection_name)
            if not sensors_view_layer.exclude:
                context.view_layer.objects.active = empty
                empty.select_set(True)
            
            empty.location = context.scene.cursor.location
        elif context.scene.sensor_type == 'FEMTOBOLT':
            base_name = "FemtoBolt"
            index = 1
            while f"{base_name}.{index:03}" in bpy.data.objects:
                index += 1
            empty_name = f"{base_name}.{index:03}"
            
            # Create an empty
            empty = bpy.data.objects.new(name=empty_name, object_data=None)
            empty.empty_display_type = 'PLAIN_AXES'
            target_collection.objects.link(empty)
            empty.empty_display_size = 0.2

            # Import the camera model
            bpy.ops.wm.obj_import(filepath= str(ROOT_DIR / 'data/FemtoBoltCasing.obj'), check_existing=True)
            imported_object = bpy.context.selected_objects[0]
            imported_object.parent = empty
            imported_object.name = f"{base_name}Casing"

            # Create camera
            cam_data = self.new_femtobolt_camera()
            cam_obj = bpy.data.objects.new(name=f"{base_name}_{index:03}_Camera", 
                                                object_data=cam_data)
            cam_obj.parent = empty
            cam_obj.location = (0.03, 0.0, -0.025)
            cam_obj.scale = (0.2, 0.2, 0.2)
            self.add_camera_properties(cam_obj, rgb=True, ir=False)
            target_collection.objects.link(cam_obj)

            empty.location = context.scene.cursor.location
        # TODO: add Kinect
        return {'FINISHED'}

class VIEW3D_OT_GeneratePose(bpy.types.Operator):
    """"""
    bl_idname = "view3d.generate_pose"
    bl_label = "Generate Random Pose"
    bl_description="Generate a rondom pose"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            return (context.scene.armature_ref)
        except: return False

    def execute(self, context):
        armature = context.scene.armature_ref
        curr_hide = armature.hide_get()
        armature.hide_set(False)
        active_curr = context.view_layer.objects.active
        context.view_layer.objects.active = armature
        current_mode = context.object.mode
        bpy.ops.object.mode_set(mode='POSE')
        bone_collection = armature.data.collections.get(context.scene.selected_bone_collection)
        for bone in armature.pose.bones:
            if bone_collection and bone.name not in bone_collection.bones:
                continue
            limit_rot = next((c for c in bone.constraints if c.type == 'LIMIT_ROTATION'), None)
            min_x = -math.pi
            max_x = math.pi
            min_y = -math.pi
            max_y = math.pi
            min_z = -math.pi
            max_z = math.pi
            if limit_rot:
                if limit_rot.use_limit_x:
                    min_x = limit_rot.min_x
                    max_x = limit_rot.max_x
                if limit_rot.use_limit_y:
                    min_y = limit_rot.min_y
                    max_y = limit_rot.max_y
                if limit_rot.use_limit_z:
                    min_z = limit_rot.min_z
                    max_z = limit_rot.max_z
            bone.rotation_mode = 'XYZ'
            bone.rotation_euler = (
                random.uniform(min_x, max_x),
                random.uniform(min_y, max_y),
                random.uniform(min_z, max_z))
        bpy.ops.object.mode_set(mode=current_mode)
        context.view_layer.objects.active = active_curr
        armature.hide_set(curr_hide)

        # Apply corrective poseshapes
        if context.scene.use_poseshapes:
            bpy.ops.view3d.pose_shapes('EXEC_DEFAULT')

        return {'FINISHED'}

class VIEW3D_OT_AttachObject(bpy.types.Operator):
    """"""
    bl_idname = "view3d.attach_object"
    bl_label = "Attach Visibility"
    bl_description="Attach the object's visibility to the selected pose"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            return (context.scene.pose_attachment) and (context.scene.pose_selection != 'NONE')
        except: return False

    def execute(self, context):
        data = []
        obj = context.scene.pose_attachment.name
        pose_path = Path(bpy.path.abspath(context.scene.pose_path))
        if pose_path.is_file():
            with open(pose_path, 'r') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    return {'CANCELLED'}
        else:
            self.report({'ERROR'}, "A file with saved poses is missing")
            return {'CANCELLED'}
        for pose in data:
            if pose['name'] == context.scene.pose_selection:
                if pose.get('render_with') is None:
                    pose['render_with'] = []
                if not obj in pose['render_with']:
                    pose['render_with'].append(obj)
                break
        with open(pose_path, 'w') as f:
            json.dump(data, f, indent=4)
        load_vis_att_from_file(self, context)
        context.scene.pose_attachment = None
        return {'FINISHED'}

class VIEW3D_OT_DetachObject(bpy.types.Operator):
    """"""
    bl_idname = "view3d.detach_object"
    bl_label = "Detach"
    bl_description="Detach the object's visibility from the selected pose"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            return (context.scene.list_attachments != "NONE")
        except: return False

    def execute(self, context):
        data = []
        obj = context.scene.list_attachments
        pose_path = Path(bpy.path.abspath(context.scene.pose_path))
        if pose_path.is_file():
            with open(pose_path, 'r') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    return {'CANCELLED'}
        else:
            self.report({'ERROR'}, "A file with saved poses is missing")
            return {'CANCELLED'}
        for pose in data:
            if pose['name'] == context.scene.pose_selection:
                if pose.get('render_with') is None:
                    break
                if obj in pose['render_with']:
                    pose['render_with'] = [att for att in pose['render_with'] if att != obj]
                break
        with open(pose_path, 'w') as f:
            json.dump(data, f, indent=4)
        load_vis_att_from_file(self, context)
        return {'FINISHED'}

class VIEW3D_OT_LoadPoses(bpy.types.Operator):
    """"""
    bl_idname = "view3d.load_poses"
    bl_label = "Load Poses"
    bl_description="Load saved/predefined poses"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        load_poses_from_file(self, context)
        load_vis_att_from_file(self, context)
        return {'FINISHED'}

class VIEW3D_OT_SavePose(bpy.types.Operator):
    """"""
    bl_idname = "view3d.save_pose"
    bl_label = "Save Pose"
    bl_description="Save the armature's current pose"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            return (context.scene.armature_ref) and (context.scene.pose_name)
        except: return False

    def get_bone_data(self, bone):
        bone_info = {}
        bone_info["name"] = bone.name
        bone.rotation_mode = 'QUATERNION'
        bone_info["rotation_quaternion"] = list(bone.rotation_quaternion)
        return bone_info

    def get_armature_data(self, context):
        active_curr = context.view_layer.objects.active
        current_mode = context.object.mode
        armature = context.scene.armature_ref
        context.view_layer.objects.active = armature
        bpy.ops.object.mode_set(mode='POSE')
        bone_collection = armature.data.collections.get(context.scene.selected_bone_collection)
        armature_info = {}
        armature_info["name"] = context.scene.pose_name
        armature_info["arm_name"] = armature.name
        armature_info["bone_col"] = bone_collection.name
        armature_info["joints"] = []
        for bone in armature.pose.bones:
            if bone.name not in bone_collection.bones:
                continue
            bone_info = self.get_bone_data(bone)
            armature_info["joints"].append(bone_info)
        bpy.ops.object.mode_set(mode=current_mode)
        context.view_layer.objects.active = active_curr
        return armature_info

    def execute(self, context):
        armature_info = self.get_armature_data(context)
        armature_info['render_with'] = []
        data = []
        pose_path = Path(bpy.path.abspath(context.scene.pose_path))
        if pose_path.is_file():
            with open(pose_path, 'r') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    return {'CANCELLED'}
        elif pose_path.is_dir():
            pose_path = pose_path / "saved_poses.json"
        else:
            self.report({'ERROR'}, "Invalid path for saved poses")
            return {'CANCELLED'}
        
        if any(entry.get("name") == armature_info.get("name") 
                and entry.get("arm_name") == context.scene.armature_ref.name 
                and entry.get("bone_col") == context.scene.selected_bone_collection for entry in data):
            self.report({'ERROR'}, "A pose with this name already exists")
            return {'CANCELLED'}
        data.append(armature_info)
        with open(pose_path, 'w') as f:
            json.dump(data, f, indent=4)
        pose_name = context.scene.pose_name
        context.scene.pose_name = ""
        context.scene.pose_path = str((pose_path))
        load_poses_from_file(self, context)
        context.scene.pose_selection = pose_name
        return {'FINISHED'}

class VIEW3D_OT_RenamePose(bpy.types.Operator):
    """"""
    bl_idname = "view3d.rename_pose"
    bl_label = "Rename"
    bl_description="Rename currently selected pose"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            return (context.scene.pose_name) and (context.scene.pose_selection != "NONE")
        except: return False

    def execute(self, context):
        # armature_info = self.get_armature_data(context)
        data = []
        pose_path = Path(bpy.path.abspath(context.scene.pose_path))
        if pose_path.is_file():
            with open(pose_path, 'r') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    return {'CANCELLED'}
        else:
            self.report({'ERROR'}, "A file with saved poses is missing")
            return {'CANCELLED'}
        if any(entry.get("name") == context.scene.pose_name 
               and entry.get("arm_name") == context.scene.armature_ref.name 
               and entry.get("bone_col") == context.scene.selected_bone_collection for entry in data):
            self.report({'ERROR'}, "A pose with this name already exists")
            return {'CANCELLED'}
        for pose in data:
            if pose['name'] == context.scene.pose_selection:
                pose['name'] = context.scene.pose_name
                break
        with open(pose_path, 'w') as f:
            json.dump(data, f, indent=4)
        context.scene.pose_name = ""
        load_poses_from_file(self, context)
        return {'FINISHED'}

class VIEW3D_OT_ApplyPose(bpy.types.Operator):
    """"""
    bl_idname = "view3d.apply_pose"
    bl_label = "Aplly Pose"
    bl_description="Apply predefined pose to the armature"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            return (context.scene.armature_ref) and (context.scene.pose_selection != "NONE")
        except: return False

    def execute(self, context):
        data = []
        pose_path = Path(bpy.path.abspath(context.scene.pose_path))
        if pose_path.is_file():
            with open(pose_path, 'r') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    return {'CANCELLED'}
        else:
            self.report({'ERROR'}, "A file with saved poses is missing")
            return {'CANCELLED'}
        armature = context.scene.armature_ref
        pose = context.scene.pose_selection
        active_curr = context.view_layer.objects.active
        current_mode = context.object.mode
        # Search for the pose
        for entry in data:
            if (entry.get("name") == pose 
                and entry.get("arm_name") == context.scene.armature_ref.name 
                and entry.get("bone_col") == context.scene.selected_bone_collection):
                context.view_layer.objects.active = armature
                bpy.ops.object.mode_set(mode='POSE')
                # Search for the bone
                for bone in armature.pose.bones:
                    for joint in entry["joints"]:
                        if joint.get("name") == bone.name:
                            # Apply bone rotation
                            bone.rotation_mode = 'QUATERNION'
                            bone.rotation_quaternion = joint["rotation_quaternion"]
                            break
                break

        bpy.ops.object.mode_set(mode=current_mode)
        context.view_layer.objects.active = active_curr
        
        # Apply corrective poseshapes
        if context.scene.use_poseshapes:
            bpy.ops.view3d.pose_shapes('EXEC_DEFAULT')
        
        return {'FINISHED'}

class VIEW3D_OT_DeletePose(bpy.types.Operator):
    """"""
    bl_idname = "view3d.delete_pose"
    bl_label = "Delete"
    bl_description="Delete currently selected pose"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            return (context.scene.pose_selection != "NONE")
        except: return False

    def execute(self, context):
        data = []
        pose_path = Path(bpy.path.abspath(context.scene.pose_path))
        if pose_path.is_file():
            with open(pose_path, 'r') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    return {'CANCELLED'}
        else:
            self.report({'ERROR'}, "A file with saved poses is missing")
            return {'CANCELLED'}
        # Exclude the pose
        new_data = [entry for entry in data if not (entry.get("name") == context.scene.pose_selection
                                                    and entry.get("arm_name") == context.scene.armature_ref.name 
                                                    and entry.get("bone_col") == context.scene.selected_bone_collection)]
        # Save updated list back
        with open(pose_path, 'w') as f:
            json.dump(new_data, f, indent=4)
        load_poses_from_file(self, context)
        return {'FINISHED'}

# Based on SMPL-X add-on implementation
def rodrigues_from_pose(armature, bone_name):
    # Use quaternion mode for all bone rotations
    if armature.pose.bones[bone_name].rotation_mode != 'QUATERNION':
        armature.pose.bones[bone_name].rotation_mode = 'QUATERNION'

    quat = armature.pose.bones[bone_name].rotation_quaternion
    (axis, angle) = quat.to_axis_angle()
    rodrigues = axis
    rodrigues.normalize()
    rodrigues = rodrigues * angle
    return rodrigues

class VIEW3D_OT_PoseShapes(bpy.types.Operator):
    bl_idname = "view3d.pose_shapes"
    bl_label = "Update Pose Shapes"
    bl_description = ("Update corrective pose shapes for current mesh of the selectd armature")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            # Enable button only if mesh is active object and parent is armature
            return (context.scene.armature_ref)
        except: return False

    # https://github.com/gulvarol/surreal/blob/master/datageneration/main_part1.py
    # Computes rotation matrix through Rodrigues formula as in cv2.Rodrigues
    def rodrigues_to_mat(self, rotvec):
        theta = np.linalg.norm(rotvec)
        r = (rotvec/theta).reshape(3, 1) if theta > 0. else rotvec
        cost = np.cos(theta)
        mat = np.asarray([[0, -r[2], r[1]],
                        [r[2], 0, -r[0]],
                        [-r[1], r[0], 0]], dtype=object)
        return(cost*np.eye(3) + (1-cost)*r.dot(r.T) + np.sin(theta)*mat)

    # https://github.com/gulvarol/surreal/blob/master/datageneration/main_part1.py
    # Calculate weights of pose corrective blend shapes
    # Input is pose of all 16 joints, output is weights for all joints except wrist
    def rodrigues_to_posecorrective_weight(self, pose, num_joints):
        joints_posecorrective = num_joints # MANO joints excluding wrist
        rod_rots = np.asarray(pose).reshape(joints_posecorrective, 3)
        mat_rots = [self.rodrigues_to_mat(rod_rot) for rod_rot in rod_rots]
        bshapes = np.concatenate([(mat_rot - np.eye(3)).ravel() for mat_rot in mat_rots[1:]])
        return(bshapes)
    
    def execute(self, context):
        armature = context.scene.armature_ref
        mesh_right = None
        mesh_left = None
        mesh_smplx = None
        for child in armature.children:
            if child.type == 'MESH' and child.data.shape_keys:
                if child.vertex_groups.get("MANO_RIGHT_HAND"):
                    mesh_right = child
                if child.vertex_groups.get("MANO_LEFT_HAND"):
                    mesh_left = child
                if child.name.startswith('SMPLX-mesh'):
                    mesh_smplx = child
        
        if mesh_smplx:
            pose = [0.0] * (NUM_SMPLX_JOINTS * 3)

            for index in range(NUM_SMPLX_JOINTS):
                joint_name = SMPLX_JOINT_NAMES[index]
                joint_pose = rodrigues_from_pose(armature, joint_name)
                pose[index*3 + 0] = joint_pose[0]
                pose[index*3 + 1] = joint_pose[1]
                pose[index*3 + 2] = joint_pose[2]

            poseweights = self.rodrigues_to_posecorrective_weight(pose, NUM_SMPLX_JOINTS)

            # Set weights for pose corrective shape keys
            for index, weight in enumerate(poseweights):
                mesh_smplx.data.shape_keys.key_blocks["Pose%03d" % index].value = weight

        if mesh_right:
            # Get armature pose in rodrigues representation
            pose = [0.0] * (NUM_MANO_JOINTS * 3)
            
            for index in range(NUM_MANO_JOINTS):
                joint_name = "RIGHT_" + lm.BONE_NAMES[index]
                joint_pose = rodrigues_from_pose(armature, joint_name)
                pose[index*3 + 0] = joint_pose[0]
                pose[index*3 + 1] = joint_pose[1]
                pose[index*3 + 2] = joint_pose[2]
            
            poseweights = self.rodrigues_to_posecorrective_weight(pose, NUM_MANO_JOINTS)

            # Set weights for pose corrective shape keys
            for index, weight in enumerate(poseweights):
                mesh_right.data.shape_keys.key_blocks[f"MANOPoseRIGHT_{index+1}"].value = weight
        
        if mesh_left:
            # Get armature pose in rodrigues representation
            pose = [0.0] * (NUM_MANO_JOINTS * 3)
            
            for index in range(NUM_MANO_JOINTS):
                joint_name = "LEFT_" + lm.BONE_NAMES[index]
                joint_pose = rodrigues_from_pose(armature, joint_name)
                pose[index*3 + 0] = joint_pose[0]
                pose[index*3 + 1] = joint_pose[1]
                pose[index*3 + 2] = joint_pose[2]
            
            poseweights = self.rodrigues_to_posecorrective_weight(pose, NUM_MANO_JOINTS)

            # Set weights for pose corrective shape keys
            for index, weight in enumerate(poseweights):
                mesh_left.data.shape_keys.key_blocks[f"MANOPoseRIGHT_{index+1}"].value = weight

        return {'FINISHED'}

class VIEW3D_OT_LoadShapes(bpy.types.Operator):
    """"""
    bl_idname = "view3d.load_shapes"
    bl_label = "Load Shapes"
    bl_description="Load saved/predefined shapes"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        load_shapes_from_file(self, context)
        return {'FINISHED'}

class VIEW3D_OT_SaveShape(bpy.types.Operator):
    """"""
    bl_idname = "view3d.save_shape"
    bl_label = "Save Shape"
    bl_description="Save the hand's current shape"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            return (context.scene.shape_name) and (
                (context.scene.deformable_mesh_right_ref) or (context.scene.deformable_mesh_left_ref))
        except: return False

    def get_shape_data(self, context):
        shape_info = {}
        shape_info["name"] = context.scene.shape_name
        mano_data = []
        mesh_right = context.scene.deformable_mesh_right_ref
        mesh_left = context.scene.deformable_mesh_left_ref
        if mesh_right:
            mano_data = [key.value for key in mesh_right.data.shape_keys.key_blocks if key.name.startswith('MANOShapeRIGHT_')]
        else:
            mano_data = [key.value for key in mesh_left.data.shape_keys.key_blocks if key.name.startswith('MANOShapeLEFT_')]
        shape_info["shape"] = mano_data
        return shape_info

    def execute(self, context):
        shape_info = self.get_shape_data(context)
        data = []
        shape_path = Path(bpy.path.abspath(context.scene.shape_path))
        if shape_path.is_file():
            with open(shape_path, 'r') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    return {'CANCELLED'}
        elif shape_path.is_dir():
            shape_path = shape_path / "saved_shapes.json"
        else:
            self.report({'ERROR'}, "Invalid path for saved shapes")
            return {'CANCELLED'}
        
        if any(entry.get("name") == context.scene.shape_name  for entry in data):
            self.report({'ERROR'}, "A shape with this name already exists")
            return {'CANCELLED'}
        data.append(shape_info)
        with open(shape_path, 'w') as f:
            json.dump(data, f, indent=4)
        shape_name = context.scene.shape_name
        context.scene.shape_name = ""
        context.scene.shape_path = str((shape_path))
        load_shapes_from_file(self, context)
        context.scene.shape_selection = shape_name
        return {'FINISHED'}

class VIEW3D_OT_RenameShape(bpy.types.Operator):
    """"""
    bl_idname = "view3d.rename_shape"
    bl_label = "Rename"
    bl_description="Rename currently selected shape"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            return (context.scene.shape_name) and (context.scene.shape_selection != "NONE")
        except: return False

    def execute(self, context):
        data = []
        shape_path = Path(bpy.path.abspath(context.scene.shape_path))
        if shape_path.is_file():
            with open(shape_path, 'r') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    return {'CANCELLED'}
        else:
            self.report({'ERROR'}, "A file with saved shapes is missing")
            return {'CANCELLED'}
        if any(entry.get("name") == context.scene.shape_name for entry in data):
            self.report({'ERROR'}, "A shape with this name already exists")
            return {'CANCELLED'}
        for shape in data:
            if shape['name'] == context.scene.shape_selection:
                shape['name'] = context.scene.shape_name
                break
        with open(shape_path, 'w') as f:
            json.dump(data, f, indent=4)
        context.scene.shape_name = ""
        load_shapes_from_file(self, context)
        return {'FINISHED'}

class VIEW3D_OT_ApplyShape(bpy.types.Operator):
    """"""
    bl_idname = "view3d.apply_shape"
    bl_label = "Aplly Shape"
    bl_description="Apply predefined shape to the armature"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            return ((context.scene.deformable_mesh_right_ref) or (
                    context.scene.deformable_mesh_left_ref)) and (context.scene.shape_selection != "NONE")
        except: return False

    def execute(self, context):
        data = []
        shape_path = Path(bpy.path.abspath(context.scene.shape_path))
        if shape_path.is_file():
            with open(shape_path, 'r') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    return {'CANCELLED'}
        else:
            self.report({'ERROR'}, "A file with saved shapes is missing")
            return {'CANCELLED'}
        shape = context.scene.shape_selection
        mesh_right = context.scene.deformable_mesh_right_ref
        mesh_left = context.scene.deformable_mesh_left_ref
        # Search for the shape
        for entry in data:
            if (entry.get("name") == shape):
                shapekeys = entry.get("shape")
                # Apply the shapekeys
                if mesh_right:
                    right_hand_shape_keys = [key for key in mesh_right.data.shape_keys.key_blocks if key.name.startswith('MANOShapeRIGHT_')]
                    for i, key_right in enumerate(right_hand_shape_keys):
                        key_right.value = shapekeys[i]
                if mesh_left:
                    left_hand_shape_keys = [key for key in mesh_left.data.shape_keys.key_blocks if key.name.startswith('MANOShapeLEFT_')]
                    for i, key_left in enumerate(left_hand_shape_keys):
                        key_left.value = shapekeys[i]
                bpy.ops.view3d.update_joint_positions('EXEC_DEFAULT')
                break
        return {'FINISHED'}

class VIEW3D_OT_DeleteShape(bpy.types.Operator):
    """"""
    bl_idname = "view3d.delete_shape"
    bl_label = "Delete"
    bl_description="Delete currently selected shape"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            return (context.scene.shape_selection != "NONE")
        except: return False

    def execute(self, context):
        data = []
        shape_path = Path(bpy.path.abspath(context.scene.shape_path))
        if shape_path.is_file():
            with open(shape_path, 'r') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    return {'CANCELLED'}
        else:
            self.report({'ERROR'}, "A file with saved shapes is missing")
            return {'CANCELLED'}
        # Exclude the shape
        new_data = [entry for entry in data if entry.get("name") != context.scene.shape_selection]
        # Save updated list back
        with open(shape_path, 'w') as f:
            json.dump(new_data, f, indent=4)
        load_shapes_from_file(self, context)
        return {'FINISHED'}

class VIEW3D_OT_ArmatureKeyframe(bpy.types.Operator):
    """"""
    bl_idname = "view3d.armature_keyframe"
    bl_label = "Keyframe"
    bl_description = "Keyframe the given armature"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            return (context.scene.armature_ref)
        except: return False

    def execute(self, context):
        armature = context.scene.armature_ref
        # bone_collection = armature.data.collections.get(context.scene.selected_bone_collection)
        active_curr = context.view_layer.objects.active
        curr_hide = armature.hide_get()
        armature.hide_set(False)
        context.view_layer.objects.active = armature
        current_mode = context.object.mode
        bpy.ops.object.mode_set(mode='POSE')
        for bone in armature.pose.bones:
            # if bone_collection and bone.name not in bone_collection.bones:
            #     continue
            bone.keyframe_insert(data_path="location")
            bone.rotation_mode = 'QUATERNION'
            bone.keyframe_insert(data_path="rotation_quaternion")
        current_frame = context.scene.frame_current
        if context.scene.keyframe_attachments == True:
            for item in _cached_pose_attachments:
                if item[0] == 'NONE':
                    continue
                if not item[0] in bpy.data.objects:
                    self.report({'ERROR'}, f'"{item[0]}" object was not found')
                    continue
                obj = bpy.data.objects[item[0]]
                # if obj is not None:
                obj.hide_render = False
                obj.keyframe_insert(data_path="hide_render")
                context.scene.frame_set(current_frame-1)
                if not is_hide_render_keyframed(obj):
                    obj.hide_render = True
                    obj.keyframe_insert(data_path="hide_render")
                context.scene.frame_set(current_frame+1)
                if not is_hide_render_keyframed(obj):
                    obj.hide_render = True
                    obj.keyframe_insert(data_path="hide_render")
                    
            context.scene.frame_set(current_frame)

        bpy.ops.object.mode_set(mode=current_mode)
        context.view_layer.objects.active = active_curr
        armature.hide_set(curr_hide)
        return {'FINISHED'}

class VIEW3D_OT_ResetPose(bpy.types.Operator):
    """"""
    bl_idname = "view3d.reset_pose"
    bl_label = "Reset"
    bl_description="Reset the pose"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            return (context.scene.armature_ref)
        except: return False

    def execute(self, context):
        armature = context.scene.armature_ref
        bone_collection = armature.data.collections.get(context.scene.selected_bone_collection)
        active_curr = context.view_layer.objects.active
        curr_hide = armature.hide_get()
        armature.hide_set(False)
        context.view_layer.objects.active = armature
        current_mode = context.object.mode
        bpy.ops.object.mode_set(mode='POSE')
        for bone in armature.pose.bones:
            if bone_collection and bone.name not in bone_collection.bones:
                continue
            bone.rotation_mode = 'XYZ'
            bone.rotation_euler = (0,0,0)
        bpy.ops.object.mode_set(mode=current_mode)
        context.view_layer.objects.active = active_curr
        armature.hide_set(curr_hide)
        return {'FINISHED'}

def export_get_bone_colllections(self, context):
    items = [("ALL", "All", "Use the whole armature")]
    armature = self.arm_ref
    if armature and armature.type == 'ARMATURE':
        items.extend([(bc.name, bc.name, f"Export {bc.name} collection") for bc in armature.data.collections])
    return items

class ExportArmatureGroup(bpy.types.PropertyGroup):
    arm_ref: bpy.props.PointerProperty(
        name="",
        type=bpy.types.Object,
        description="Pick an armature to export",
        poll=lambda self, obj: obj.type == 'ARMATURE',
    ) # type: ignore

    bone_col: bpy.props.EnumProperty(
        name="",
        description="Bone collection",
        items=export_get_bone_colllections,
    ) # type: ignore

class VIEW3D_OT_AddExportArmature(bpy.types.Operator):
    bl_idname = "view3d.add_export_armature"
    bl_label = "Add"
    bl_description = "Add an armature to export"

    def execute(self, context):
        scene = context.scene
        scene.export_arm.add()
        return {'FINISHED'}

class VIEW3D_OT_RemoveExportArmature(bpy.types.Operator):
    bl_idname = "view3d.remove_export_item"
    bl_label = ""
    bl_description = "Remove an armature from export"

    index: bpy.props.IntProperty() # type: ignore

    def execute(self, context):
        scene = context.scene
        if 0 <= self.index < len(scene.export_arm):
            scene.export_arm.remove(self.index)
        return {'FINISHED'}

class PoseArmatureGroup(bpy.types.PropertyGroup):
    arm_ref: bpy.props.PointerProperty(
        name="",
        type=bpy.types.Object,
        description="Pick an armature to sample poses from",
        poll=lambda self, obj: obj.type == 'ARMATURE',
    ) # type: ignore

    bone_col: bpy.props.EnumProperty(
        name="",
        description="Bone collection",
        items=export_get_bone_colllections,
    ) # type: ignore

    group: bpy.props.IntProperty(
        name="",
        description="Poses in the same group are keyframed on the same frame. " \
        "Each frame will contain a combination of those poses.\n" \
        "Avoid assigning bone collections with shared bones to the same group",
        default=1,
        min=1,
        max=100,
        soft_max=10,
    ) # type: ignore

class VIEW3D_OT_AddPoseArmature(bpy.types.Operator):
    bl_idname = "view3d.add_pose_armature"
    bl_label = "Add"
    bl_description = "Add an armature for keyframe generation"

    def execute(self, context):
        scene = context.scene
        scene.pose_arm.add()
        return {'FINISHED'}

class VIEW3D_OT_RemovePoseArmature(bpy.types.Operator):
    bl_idname = "view3d.remove_pose_item"
    bl_label = ""
    bl_description = "Remove an armature from keyframe generation"

    index: bpy.props.IntProperty() # type: ignore

    def execute(self, context):
        scene = context.scene
        if 0 <= self.index < len(scene.pose_arm):
            scene.pose_arm.remove(self.index)
        return {'FINISHED'}

class VIEW3D_OT_GenerateFrames(bpy.types.Operator):
    """"""
    bl_idname = "view3d.generate_frames"
    bl_label = "Generate Keyframes"
    bl_description="Generate keyframes with the selected pool of poses.\n" \
    "If you are regenerating the keyframes clear the armature animation data first"

    i: int

    def keyframe(self, context, pose_group, index=0):
        # If we reached the end of the group
        if index == len(pose_group):
            # Move to the next frame
            self.i += 1
            return
        # Recursively iterate over all poses to get their combinations
        (arm, poses), = pose_group[index].items()
        for pose in poses:
            context.scene.armature_ref = arm.arm_ref
            context.scene.selected_bone_collection = arm.bone_col
            context.scene.pose_selection = pose
            context.scene.frame_set(self.i)
            bpy.ops.view3d.apply_pose('EXEC_DEFAULT')
            bpy.ops.view3d.armature_keyframe('EXEC_DEFAULT')
            self.keyframe(context, pose_group, index + 1)

    def execute(self, context):
        pose_path = Path(bpy.path.abspath(context.scene.pose_path))
        if pose_path.is_file():
            with open(pose_path, 'r') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    return {'CANCELLED'}
        else:
            self.report({'ERROR'}, "A file with saved poses is missing.\nSave a pose to create one")
            return {'CANCELLED'}

        pose_arms = context.scene.pose_arm
        arm_groups = {}
        # Sort by groups
        if len(pose_arms) > 0:
            for pose_arm in pose_arms:
                poses = [pose["name"] for pose in data
                    if pose["arm_name"] == pose_arm.arm_ref.name
                    and pose["bone_col"] == pose_arm.bone_col]
                if pose_arm.group in arm_groups:
                    arm_groups[pose_arm.group].append({pose_arm: poses})
                else:
                    arm_groups[pose_arm.group] = [{pose_arm: poses}]
        
        current_frame = context.scene.frame_current
        self.i = current_frame

        for arm_group in arm_groups:
            self.keyframe(context, arm_groups[arm_group])
        
        return {'FINISHED'}

class VIEW3D_OT_ExportMetadata(bpy.types.Operator):
    """"""
    bl_idname = "view3d.export_metadata"
    bl_label = "Export Metadata"
    bl_description="Export sensor and hand metadata"

    def get_sensors_data(self, context, sensor_collection, matrix_world):
        # TODO: detect hand visibility, bool "{armature bone_col} in frame"
        sensors_data = []
        for obj in sensor_collection.objects:
            if obj.type == 'CAMERA':
                location = matrix_world @ obj.matrix_world.to_translation()
                rotation = obj.matrix_world.to_quaternion()
                sens_info = {
                    "name": obj.name,
                    "location": list(location),
                    "rotation_quaternion": list(rotation),
                }
                sensors_data.append(sens_info)
        return sensors_data

    def get_bone_data(self, bone, matrix_world, armature_world):
        location = matrix_world @ (armature_world @ bone.head)
        bone_info = {}
        bone_info["name"] = bone.name
        bone_info["location"] = list(location)
        bone.rotation_mode = 'QUATERNION'
        bone_info["rotation_quaternion"] = list((armature_world @ bone.matrix).to_quaternion())
        return bone_info

    def get_armature_data(self, context, export_armatures, matrix_world):
        armature_data = []
        active_curr = context.view_layer.objects.active
        current_mode = context.object.mode
        for export in export_armatures:
            armature = export.arm_ref
            if not armature:
                continue
            bone_collection = armature.data.collections.get(export.bone_col)
            context.view_layer.objects.active = armature
            bpy.ops.object.mode_set(mode='POSE')
            armature_info = {}
            armature_info["name"] = armature.name
            armature_info["bone_collection"] = export.bone_col
            armature_info["joints"] = []
            # context.view_layer.objects.active = armature
            for bone in armature.pose.bones:
                if bone_collection and bone.name not in bone_collection.bones:
                    continue
                bone_info = self.get_bone_data(bone, matrix_world, armature.matrix_world)
                armature_info["joints"].append(bone_info)
            armature_data.append(armature_info)
        bpy.ops.object.mode_set(mode=current_mode)
        context.view_layer.objects.active = active_curr
        return armature_data

    def get_mano_shape_data(self, context, export_armatures):
        mano_data = []
        for export in export_armatures:
            armature = export.arm_ref
            if not armature:
                continue
            shape_info = {}
            shape_info["armature"] = armature.name
            shape_info["right_hand_shape"] = []
            shape_info["left_hand_shape"] = []
            for obj in armature.children:
                if obj.type == 'MESH' and obj.data.shape_keys:
                    if obj.vertex_groups["MANO_RIGHT_HAND"]:
                        shape_info["right_hand_shape"] = [key.value for key in obj.data.shape_keys.key_blocks if key.name.startswith('MANOShapeRIGHT_')]
                    elif obj.vertex_groups["MANO_LEFT_HAND"]:
                        shape_info["left_hand_shape"] = [key.value for key in obj.data.shape_keys.key_blocks if key.name.startswith('MANOShapeLEFT_')]
            mano_data.append(shape_info)
        return mano_data

    def execute(self, context):
        # Check for save folder
        save_folder = context.scene.save_folder
        if not save_folder:
            self.report({'ERROR'}, "No save folder provided")
            return {'CANCELLED'}
        # Create metadata folder if it doesn't exists
        meta_path = Path(bpy.path.abspath(save_folder), "metadata")
        meta_path.mkdir(exist_ok=True)

        sensor_collection = bpy.data.collections.get('Sensors')
        export_armatures = context.scene.export_arm
        # Get the reference frame
        origin = context.scene.origin_ref
        if origin:
            rot = origin.rotation_euler.to_matrix().to_4x4()
            trans = Matrix.Translation(origin.location)
            matrix_world = (trans @ rot).inverted()
        else:
            matrix_world = Matrix.Translation(Vector((0,0,0)))
        # Iterate over all frames
        current_frame = context.scene.frame_current
        for i in range(context.scene.frame_start, context.scene.frame_end+1):
            context.scene.frame_set(i)
            mano_data = self.get_mano_shape_data(context, export_armatures)
            # Get sensor(s) metadata
            sensors_data = self.get_sensors_data(context, sensor_collection, matrix_world) if sensor_collection else []
            # Get armature(s) metadata
            armature_data = self.get_armature_data(context, export_armatures, matrix_world) if len(export_armatures) > 0 else []
            # Save metadata
            export_filepath = Path(meta_path, f"Frame_{i:06}.json")
            with open(export_filepath, 'w') as f:
                json.dump({"Sensor data":sensors_data, "MANO hand data":mano_data, "Armature data":armature_data}, f, indent=4)
        context.scene.frame_set(current_frame)
        self.report({'INFO'}, "Metadata successfully exported")
        return {'FINISHED'}

class VIEW3D_OT_InfoBox(bpy.types.Operator):
    bl_idname = "view3d.info_box"
    bl_label = ""
    bl_description = "It is recomended to use a rendering script.\n" \
    "Make sure that the 'Stereoscopy' property is checked and only the necessary cameras are selected.\n" \
    "Rendering the animation inside the Blender GUI will freeze the application"
    
    def execute(self, context):
        self.report({'INFO'}, "It is recomended to use a rendering script. " \
        "Make sure that the 'Stereoscopy' property is checked and only the necessary cameras are selected.\n" \
        "Rendering the animation inside the Blender GUI will freeze the application.")
        return {'FINISHED'}

class VIEW3D_OT_MoveSensorToOrigin(bpy.types.Operator):
    bl_idname = "view3d.move_sensor_to_origin"
    bl_label = "Move to Origin"
    bl_description = "Move the sensor to the origin"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            if not context.selected_objects:
                return False
            # Check the "Sensors" collection
            sensors_coll = bpy.data.collections.get('Sensors')
            if not sensors_coll:
                return False
            # Check every selected object
            for obj in context.selected_objects:
                if obj.type != 'EMPTY':
                    return False
                if obj.name not in sensors_coll.objects:
                    return False
            return True
        except: return False
    
    def execute(self, context):
        sensor_list = context.selected_objects
        origin = context.scene.origin_ref
        origin_loc = origin.location if origin else (0, 0, 0)
        for sensor in sensor_list:
            sensor.location = origin_loc
        return{'FINISHED'}

class VIEW3D_OT_RandomSensorPosition(bpy.types.Operator):
    bl_idname = "view3d.random_sensor_position"
    bl_label = "Random Position"
    bl_description = "Set random sensor position on the sampling mesh.\n" \
    "Select one or multiple sensors to activate this button"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            if not context.selected_objects:
                return False
            # Check the "Sensors" collection
            sensors_coll = bpy.data.collections.get('Sensors')
            if not sensors_coll:
                return False
            # Check every selected object
            for obj in context.selected_objects:
                if obj.type != 'EMPTY':
                    return False
                if obj.name not in sensors_coll.objects:
                    return False
            # Check that scene property exists
            return bool(context.scene.random_positions_ref)
        except: return False

    def execute(self, context):
        sensor_list = context.selected_objects
        sample = context.scene.random_positions_ref
        context.view_layer.objects.active = sample
        sample_mesh = sample.data
        world_matrix = sample.matrix_world
        sample_ind = [vertex.index for vertex in sample_mesh.vertices]
        if len(sample_ind) < len(sensor_list):
            self.report({'ERROR'}, 'Selected mesh has too few vertices')
            return{'CANCELLED'}
        for sensor in sensor_list:
            rand_index = random.choice(sample_ind)
            vert_rand = sample_mesh.vertices[rand_index]
            sample_ind.remove(rand_index)
            sensor.location = world_matrix @ vert_rand.co
            orient = context.scene.sensor_orientation
            if orient == "NORMAL":
                sensor.rotation_quaternion = Vector((1.0, 0.0, 0.0, 0.0))
                from_dir = Vector((0.0, 0.0, -1.0))
                to_dir = vert_rand.normal
                rotation_quat = from_dir.rotation_difference(to_dir)
                sensor.rotation_mode = 'QUATERNION'
                sensor.rotation_quaternion = rotation_quat
            elif orient == "NEGNORMAL":
                sensor.rotation_quaternion = Vector((1.0, 0.0, 0.0, 0.0))
                from_dir = Vector((0.0, 0.0, 1.0))
                to_dir = vert_rand.normal
                rotation_quat = from_dir.rotation_difference(to_dir)
                sensor.rotation_mode = 'QUATERNION'
                sensor.rotation_quaternion = rotation_quat
            elif orient == "ORIGIN":
                sensor.rotation_quaternion = Vector((1.0, 0.0, 0.0, 0.0))
                from_dir = Vector((0.0, 0.0, -1.0))
                to_dir = sample.location - sensor.location
                rotation_quat = from_dir.rotation_difference(to_dir)
                sensor.rotation_mode = 'QUATERNION'
                sensor.rotation_quaternion = rotation_quat
            elif orient == "CURSOR":
                sensor.rotation_quaternion = Vector((1.0, 0.0, 0.0, 0.0))
                from_dir = Vector((0.0, 0.0, -1.0))
                to_dir = context.scene.cursor.location - sensor.location
                rotation_quat = from_dir.rotation_difference(to_dir)
                sensor.rotation_mode = 'QUATERNION'
                sensor.rotation_quaternion = rotation_quat
        for sensor in sensor_list:
            sensor.select_set(True)
        context.view_layer.objects.active = sensor
        return{'FINISHED'}

"""class VIEW3D_OT_RandomSensorRotation(bpy.types.Operator):
    bl_idname = "view3d.random_sensor_rotation"
    bl_label = "Random Rotation"
    bl_description = "Set random sensor rotation along its z-axis"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            return ((context.object.type == 'EMPTY') and 
                    (context.object.name in bpy.data.collections.get('Sensors').objects) and 
                    (context.scene.random_positions_ref))
        except: return False

    def execute(self, context):
        sensor = context.view_layer.objects.active
        sensor.rotation_mode = 'QUATERNION'
        angle = 360 / random.randint(1, int(context.scene.angle_restriction))
        print(angle)
        rot_quat = Quaternion((0, 0, 1), math.radians(angle))  # axis-angle: axis, angle in radians

        # Apply rotation
        sensor.rotation_quaternion = sensor.rotation_quaternion @ rot_quat
        return{'FINISHED'}"""

class VIEW3D_OT_SensorKeyframe(bpy.types.Operator):
    bl_idname = "view3d.sensor_keyframe"
    bl_label = "Keyframe"
    bl_description = "Set the current sensor postion and orientation as a keyframe"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            if not context.selected_objects:
                return False
            # Check the "Sensors" collection
            sensors_coll = bpy.data.collections.get('Sensors')
            if not sensors_coll:
                return False
            # Check every selected object
            for obj in context.selected_objects:
                if obj.type != 'EMPTY':
                    return False
                if obj.name not in sensors_coll.objects:
                    return False
            return True
        except: return False

    def execute(self, context):
        sensor_list = context.selected_objects
        for sensor in sensor_list:
            sensor.keyframe_insert(data_path="location")
            sensor.keyframe_insert(data_path="rotation_quaternion")
        return{'FINISHED'}

class VIEW3D_OT_ResetMeshShape(bpy.types.Operator):
    bl_idname = "view3d.reset_mesh_shape"
    bl_label = "Reset"
    bl_description = "Reset the shape"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            return ((context.scene.deformable_mesh_right_ref) and 
                    (context.scene.deformable_mesh_right_ref.data.shape_keys) or
                    (context.scene.deformable_mesh_left_ref) and 
                    (context.scene.deformable_mesh_left_ref.data.shape_keys))
        except: return False

    def execute(self, context):
        mesh_right = context.scene.deformable_mesh_right_ref
        if mesh_right:
            right_hand_shape_keys = [key for key in mesh_right.data.shape_keys.key_blocks if key.name.startswith('MANOShapeRIGHT_')]
            for key_right in right_hand_shape_keys:
                key_right.value = 0.0
        mesh_left = context.scene.deformable_mesh_left_ref
        if mesh_left:
            left_hand_shape_keys = [key for key in mesh_left.data.shape_keys.key_blocks if key.name.startswith('MANOShapeLEFT_')]
            for key_left in left_hand_shape_keys:
                key_left.value = 0.0
        bpy.ops.view3d.update_joint_positions('EXEC_DEFAULT')
        return{'FINISHED'}

class VIEW3D_OT_RandomMeshShape(bpy.types.Operator):
    bl_idname = "view3d.random_mesh_shape"
    bl_label = "Random Shape"
    bl_description = "Set random shape of the mesh based on its Shape Keys"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            return ((context.scene.deformable_mesh_right_ref) and 
                    (context.scene.deformable_mesh_right_ref.data.shape_keys) or
                    (context.scene.deformable_mesh_left_ref) and 
                    (context.scene.deformable_mesh_left_ref.data.shape_keys))
        except: return False

    def execute(self, context):
        mesh_right = context.scene.deformable_mesh_right_ref
        mesh_left = context.scene.deformable_mesh_left_ref
        if mesh_right and mesh_left:
            right_hand_shape_keys = [key for key in mesh_right.data.shape_keys.key_blocks if key.name.startswith('MANOShapeRIGHT_')]
            left_hand_shape_keys = [key for key in mesh_left.data.shape_keys.key_blocks if key.name.startswith('MANOShapeLEFT_')]
            if len(right_hand_shape_keys) != len(left_hand_shape_keys):
                self.report({'ERROR'}, "Meshes have different number of hand shape keys")
            for key_right, key_left in zip(right_hand_shape_keys, left_hand_shape_keys):
                key_right.value = key_left.value = random.gauss(0.0, context.scene.std_slider)
        else:
            if mesh_right:
                right_hand_shape_keys = [key for key in mesh_right.data.shape_keys.key_blocks if key.name.startswith('MANOShapeRIGHT_')]
                for key_right in right_hand_shape_keys:
                    key_right.value = random.gauss(0.0, context.scene.std_slider)
            if mesh_left:
                left_hand_shape_keys = [key for key in mesh_left.data.shape_keys.key_blocks if key.name.startswith('MANOShapeLEFT_')]
                for key_left in left_hand_shape_keys:
                    key_left.value = random.gauss(0.0, context.scene.std_slider)
        bpy.ops.view3d.update_joint_positions('EXEC_DEFAULT')
        return{'FINISHED'}

class VIEW3D_OT_ShapeKeyframe(bpy.types.Operator):
    bl_idname = "view3d.shape_keyframe"
    bl_label = "Keyframe"
    bl_description = "Set the current shape as a keyframe"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            return ((context.scene.deformable_mesh_right_ref) and 
                    (context.scene.deformable_mesh_right_ref.data.shape_keys) or
                    (context.scene.deformable_mesh_left_ref) and 
                    (context.scene.deformable_mesh_left_ref.data.shape_keys))
        except: return False

    def execute(self, context):
        mesh_right = context.scene.deformable_mesh_right_ref
        if mesh_right:
            right_hand_shape_keys = [key for key in mesh_right.data.shape_keys.key_blocks if key.name.startswith('MANOShapeRIGHT_')]
            for key in right_hand_shape_keys:
                key.keyframe_insert(data_path="value")
        mesh_left = context.scene.deformable_mesh_left_ref
        if mesh_left:
            left_hand_shape_keys = [key for key in mesh_left.data.shape_keys.key_blocks if key.name.startswith('MANOShapeLEFT_')]
            for key in left_hand_shape_keys:
                key.keyframe_insert(data_path="value")
        return{'FINISHED'}

class VIEW3D_OT_UpdateJointPositions(bpy.types.Operator):
    bl_idname = "view3d.update_joint_positions"
    bl_label = "Update Joint Positions"
    bl_description = "Update joint positions of the deformed mesh"
    bl_options = {'REGISTER', 'UNDO'}

    J_regressor_right = None
    J_regressor_left = None

    @classmethod
    def poll(cls, context):
        try:
            return ((context.scene.deformable_mesh_right_ref) and 
                    (context.scene.deformable_mesh_right_ref.data.shape_keys) and 
                    (context.scene.deformable_mesh_right_ref.parent.type == 'ARMATURE') or
                    (context.scene.deformable_mesh_left_ref) and 
                    (context.scene.deformable_mesh_left_ref.data.shape_keys) and 
                    (context.scene.deformable_mesh_left_ref.parent.type == 'ARMATURE'))
        except: return False

    def execute(self, context):
        mesh_right = context.scene.deformable_mesh_right_ref
        if mesh_right:
            # Temporarily disable the Armature modifier
            mod_show_list = []
            for mod in mesh_right.modifiers:
                if mod.type == 'ARMATURE':
                    mod_show_list.append(mod.show_viewport)
                    mod.show_viewport = False
            depsgraph = bpy.context.evaluated_depsgraph_get()
            eval_obj = mesh_right.evaluated_get(depsgraph)
            eval_mesh = eval_obj.to_mesh()
            try:
                vg_index = eval_obj.vertex_groups["MANO_RIGHT_HAND"].index
            except:
                self.report({'ERROR'}, 'No vertex group "MANO_RIGHT_HAND" was found.\nAssign hand vertices to this group.')
                return{'CANCELLED'}
            vertices = np.array([v.co[:] for v in eval_mesh.vertices if vg_index in [vg.group for vg in v.groups]])
            # Re-enable the Armature modifier
            for i, mod in enumerate(mesh_right.modifiers):
                if mod.type == 'ARMATURE':
                    mod.show_viewport = mod_show_list[i]
            # Cash the regressor and the template
            if self.J_regressor_right is None:
                mano_path = Path(bpy.path.abspath(context.scene.mano_folder) + "MANO_RIGHT.npz")
                if not mano_path.exists():
                    self.report({'ERROR'}, f"Couldn't find MANO_RIGHT.npz file.\nPlease provide a valid path in the MANO Hand panel")
                    return{'CANCELLED'}
                self.J_regressor_right = lm.load_regressor(str(mano_path))
            update_joint_positions(mesh_right.parent, self.J_regressor_right, vertices, context, "RIGHT_")
        
        mesh_left = context.scene.deformable_mesh_left_ref
        if mesh_left:
            # Temporarily disable the Armature modifier
            mod_show_list = []
            for mod in mesh_left.modifiers:
                if mod.type == 'ARMATURE':
                    mod_show_list.append(mod.show_viewport)
                    mod.show_viewport = False
            depsgraph = bpy.context.evaluated_depsgraph_get()
            eval_obj = mesh_left.evaluated_get(depsgraph)
            eval_mesh = eval_obj.to_mesh()
            try:
                vg_index = eval_obj.vertex_groups["MANO_LEFT_HAND"].index
            except:
                self.report({'ERROR'}, 'No vertex group "MANO_LEFT_HAND" was found.\nAssign hand vertices to this group.')
                return{'CANCELLED'}
            vertices = np.array([v.co[:] for v in eval_mesh.vertices if vg_index in [vg.group for vg in v.groups]])
            # Re-enable the Armature modifier
            for i, mod in enumerate(mesh_left.modifiers):
                if mod.type == 'ARMATURE':
                    mod.show_viewport = mod_show_list[i]
            # Cash the regressor and the template
            if self.J_regressor_left is None:
                mano_path = Path(bpy.path.abspath(context.scene.mano_folder) + "MANO_LEFT.npz")
                if not mano_path.exists():
                    self.report({'ERROR'}, f"Couldn't find MANO_LEFT.npz file.\nPlease provide a valid path in the MANO Hand panel")
                    return{'CANCELLED'}
                self.J_regressor_left = lm.load_regressor(str(mano_path))
            update_joint_positions(mesh_left.parent, self.J_regressor_left, vertices, context, "LEFT_")
        
        return{'FINISHED'}

class VIEW3D_OT_AddMANOHand(bpy.types.Operator):
    bl_idname = "view3d.add_mano_hand"
    bl_label = "Add"
    bl_description = "Add a MANO hand model"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.scene.mano_folder
    
    def execute(self, context):
        mano_path = Path(bpy.path.abspath(context.scene.mano_folder) + f'MANO_{context.scene.hand_selection}.npz')
        if not mano_path.exists():
            self.report({'ERROR'}, f"Couldn't find MANO_{context.scene.hand_selection}.npz file")
            return{'CANCELLED'}
        obj = lm.load_mano_hand(context.scene.hand_selection, mano_path)
        obj.location = context.scene.cursor.location
        return{'FINISHED'}

class VIEW3D_OT_ImportMANO(bpy.types.Operator):
    bl_idname = "view3d.import_mano"
    bl_label = "Load Import Script"
    bl_description = "Load the import script that will unpack mano files to be used in this add-on"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        script_name = "unpack_mano.py"
        # Remove the old script to avoid duplicates
        if script_name in bpy.data.texts:
            bpy.data.texts.remove(bpy.data.texts[script_name])
        # Load the script
        textblock = bpy.data.texts.load(str(ROOT_DIR / "scripts" / script_name))
        # Jump to Scripting workspace
        bpy.context.window.workspace = bpy.data.workspaces['Scripting']
        # Select the loaded script in a Text Editor
        def delayed_switch():
            for window in bpy.context.window_manager.windows:
                screen = window.screen
                for area in screen.areas:
                    if area.type == 'TEXT_EDITOR':
                        for space in area.spaces:
                            if space.type == 'TEXT_EDITOR':
                                space.text = textblock              # switch to the script
                                textblock.current_line_index = 0    # go to the top of the page
                                return None   # stop the timer
            return 0.05  # keep retrying until Text Editor exists

        bpy.app.timers.register(delayed_switch, first_interval=0.05)

        return{'FINISHED'}

class VIEW3D_OT_ConfigureCompositing(bpy.types.Operator):
    bl_idname = "view3d.configure_compositing"
    bl_label = "Configure Compositing"
    bl_description = "Configure the compositing node tree for sensor output simulation and depth pass"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        context.scene.use_nodes = True
        context.view_layer.use_pass_mist = True
        tree = context.scene.node_tree
        if context.scene.override_compositing:
            tree.nodes.clear()
        render_layers = tree.nodes.get("Render Layers")
        if not render_layers:
            render_layers = tree.nodes.new(type='CompositorNodeRLayers')
        composite = tree.nodes.get("Composite")
        if not composite:
            composite = tree.nodes.new(type='CompositorNodeComposite')
        # Add compositing nodes
        lens_distortion_image = tree.nodes.new(type='CompositorNodeLensdist')
        lens_distortion_alpha = tree.nodes.new(type='CompositorNodeLensdist')
        lens_distortion_mist = tree.nodes.new(type='CompositorNodeLensdist')
        invert_depth = tree.nodes.new(type='CompositorNodeInvert')
        invert_depth.invert_rgb = True
        mix_rgb = tree.nodes.new(type='CompositorNodeMixRGB')
        mix_rgb.blend_type = 'MULTIPLY'
        mix_rgb.name = "DepthImage"
        hue_correct = tree.nodes.new(type='CompositorNodeHueCorrect')
        hue_correct.name = "BlackAndWhiteFilter"
        render_layers.location = (0, 0)
        lens_distortion_image.location = (300, 350)
        lens_distortion_alpha.location = (300, 0)
        lens_distortion_mist.location = (300, -250)
        hue_correct.location = (500, 350)
        invert_depth.location = (500, -250)
        mix_rgb.location = (700, 0)
        composite.location = (1000, 0)
        # Set the distortion to 1.0
        lens_distortion_image.inputs["Distortion"].default_value = 1.0
        lens_distortion_alpha.inputs["Distortion"].default_value = 1.0
        lens_distortion_mist.inputs["Distortion"].default_value = 1.0
        # Disable the distortion for now
        lens_distortion_image.mute = True
        lens_distortion_alpha.mute = True
        lens_distortion_mist.mute = True
        # Set the saturation curve to a flat line at 0.0
        sat_curve = hue_correct.mapping.curves[1]  # 0=H, 1=S, 2=V
        # Remove existing points on the curve
        for i in reversed(range(2, len(sat_curve.points))):
            sat_curve.points.remove(sat_curve.points[i])
        sat_curve.points[0].location = (0.0, 0.0)
        sat_curve.points[1].location = (1.0, 0.0)
        # Link the nodes together
        tree.links.new(render_layers.outputs['Image'], lens_distortion_image.inputs['Image'])
        tree.links.new(lens_distortion_image.outputs['Image'], hue_correct.inputs['Image'])
        tree.links.new(render_layers.outputs['Alpha'], lens_distortion_alpha.inputs['Image'])
        tree.links.new(render_layers.outputs['Mist'], lens_distortion_mist.inputs['Image'])
        tree.links.new(lens_distortion_mist.outputs['Image'], invert_depth.inputs['Color'])
        tree.links.new(lens_distortion_alpha.outputs['Image'], mix_rgb.inputs[1])
        tree.links.new(invert_depth.outputs['Color'], mix_rgb.inputs[2])
        if context.scene.light_selection == 'DEPTH':
            tree.links.new(mix_rgb.outputs['Image'], composite.inputs['Image'])
        else:
            tree.links.new(hue_correct.outputs['Image'], composite.inputs['Image'])
            if context.scene.light_selection == 'RGB':
                hue_correct.mute = True
        if context.scene.light_selection != "DEPTH":
            context.view_layer.use_pass_mist = False
        return{'FINISHED'}

class VIEW3D_PT_Export(bpy.types.Panel):
    """"""
    bl_label = "Export Settings"
    bl_idname = "VIEW3D_PT_Export"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Hand Poser'

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        
        layout_row = layout.row(align=True)
        split_save = layout_row.split(factor=0.3, align=True)
        split_save.label(text="Save Folder:")
        split_save.prop(scene, "save_folder")

        layout.separator(type='LINE')
        layout_row = layout.row(align=True)
        split_light = layout_row.split(factor=0.3, align=True)
        split_light.label(text="Image:")
        split_light.prop(scene, "light_selection")
        layout_row = layout.row(align=True)
        split_compositing = layout_row.split(factor=0.7, align=True)
        split_compositing.operator(VIEW3D_OT_ConfigureCompositing.bl_idname)
        split_compositing.prop(scene, "override_compositing")
        
        layout.separator(type='LINE')
        layout_row = layout.row(align=True)
        split_origin = layout_row.split(factor=0.3, align=True)
        split_origin.label(text="World Origin:")
        split_origin.prop(scene, "origin_ref", icon="ORIENTATION_PARENT")

        layout_col = layout.column(align=True)
        layout_col.label(text="Armatures to export:")
        for i, item in enumerate(scene.export_arm):
            box = layout_col.box()
            box_row = box.row(align=False)
            box_col = box_row.column(align=True)
            box_row_arm = box_col.row(align=True)
            box_arm = box_row_arm.split(factor=0.27, align=True)
            box_arm.label(text="Armature:")
            box_arm.prop(item, "arm_ref", icon='ARMATURE_DATA')
            box_row_bone = box_col.row(align=True)
            box_bone = box_row_bone.split(factor=0.27, align=True)
            box_bone.label(text="Bones:")
            box_bone.prop(item, "bone_col", icon='GROUP_BONE')
            col_x = box_row.column(align=True)
            remove_op = col_x.operator("view3d.remove_export_item", icon='X')
            remove_op.index = i
        layout_col.operator("view3d.add_export_armature", icon='ADD')
        
        # layout.separator(type='LINE')

class VIEW3D_PT_MANO_Model(bpy.types.Panel):
    """"""
    bl_label = "MANO Hand Model"
    bl_idname = "VIEW3D_PT_MANO_Model"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Hand Poser'

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        
        layout_row = layout.row(align=True)
        split_mano = layout_row.split(factor=0.35, align=True)
        split_mano.label(text="MANO Folder:")
        split_mano.prop(scene, "mano_folder")
        layout.operator(VIEW3D_OT_ImportMANO.bl_idname)
        # layout_col.operator(VIEW3D_OT_ImportMANO.bl_idname)
        layout_col = layout.column(align=True)
        layout_row = layout_col.row(align=True)
        split_hand = layout_row.split(factor=0.35, align=True)
        split_hand.label(text="Hand:")
        split_hand.prop(scene, "hand_selection")
        layout_col.operator(VIEW3D_OT_AddMANOHand.bl_idname, icon='ADD')

class VIEW3D_PT_Pose(bpy.types.Panel):
    """"""
    bl_label = "Pose"
    bl_idname = "VIEW3D_PT_Pose"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Hand Poser'

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        layout_arm_col = layout.column(align=True)
        layout_row_arm = layout_arm_col.row(align=True)
        layout_arm = layout_row_arm.split(factor=0.3, align=True)
        layout_arm.label(text="Armature:")
        layout_arm.prop(scene, "armature_ref", icon='ARMATURE_DATA')
        layout_row_bone = layout_arm_col.row(align=True)
        layout_bone = layout_row_bone.split(factor=0.3, align=True)
        layout_bone.label(text="Bones:")
        layout_bone.prop(scene, "selected_bone_collection", icon='GROUP_BONE')

        layout_pose_col = layout.column(align=True)
        layout_pose_col_path = layout_pose_col.row(align=True)
        layout_pose_path = layout_pose_col_path.split(factor=0.3, align=True)
        layout_pose_path.label(text="Pose Path:")
        layout_pose_path.prop(scene, "pose_path")
        layout_pose_col.operator(VIEW3D_OT_LoadPoses.bl_idname)
        layout_pose_col = layout.column(align=True)
        layout_pose_row = layout_pose_col.row(align=True)
        layout_pose = layout_pose_row.split(factor=0.3, align=True)
        layout_pose.label(text="Pose:")
        layout_pose.prop(scene, "pose_selection")
        layout_apply_pose_row = layout_pose_col.row(align=True)
        layout_apply_pose = layout_apply_pose_row.split(factor=0.7, align=True)
        layout_apply_pose.operator(VIEW3D_OT_ApplyPose.bl_idname)
        layout_apply_pose.operator(VIEW3D_OT_DeletePose.bl_idname)
        
        layout_save_col = layout.column(align=True)
        # layout_pose_name_row = layout_save_col.row(align=True)
        # layout_pose_name = layout_pose_name_row.split(factor=0.3, align=True)
        # layout_pose_name.label(text="Pose Name:")
        layout_save_col.prop(scene, "pose_name", placeholder="Pose Name")
        layout_sr_row = layout_save_col.row(align=True)
        layout_save_rename = layout_sr_row.split(factor=0.7, align=True)
        layout_save_rename.operator(VIEW3D_OT_SavePose.bl_idname)
        layout_save_rename.operator(VIEW3D_OT_RenamePose.bl_idname)
        layout_rand_row = layout.row(align=True)
        layout_rand = layout_rand_row.split(factor=0.7, align=True)
        layout_rand.operator(VIEW3D_OT_GeneratePose.bl_idname)
        layout_rand.operator(VIEW3D_OT_ResetPose.bl_idname)
        layout_corr = layout.column(align=True)
        layout_corr.prop(scene, "use_poseshapes")
        layout_corr.operator(VIEW3D_OT_PoseShapes.bl_idname)
        layout.operator(VIEW3D_OT_ArmatureKeyframe.bl_idname)
        
        layout.separator(type='LINE')
        layout_attach_col = layout.column(align=True)
        layout_attach_col.prop(scene, "keyframe_attachments")
        layout_attach_row = layout_attach_col.row(align=True)
        layout_attach_list = layout_attach_row.split(factor=0.31, align=True)
        layout_attach_list.label(text="Attachments:")
        layout_attach_list.prop(scene, "list_attachments")
        layout_attach_col.prop(scene, "pose_attachment")
        layout_attach_action_row = layout_attach_col.row(align=True)
        layout_attach_action = layout_attach_action_row.split(factor=0.7, align=True)
        layout_attach_action.operator(VIEW3D_OT_AttachObject.bl_idname)
        layout_attach_action.operator(VIEW3D_OT_DetachObject.bl_idname)

        # layout_key = layout.column(align=True)
        # layout_key.prop(scene, "keyframe_attachments")
        # layout_key.operator(VIEW3D_OT_ArmatureKeyframe.bl_idname)

class VIEW3D_PT_Shape(bpy.types.Panel):
    """"""
    bl_label = "Shape"
    bl_idname = "VIEW3D_PT_Shape"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Hand Poser'

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        layout_col = layout.column(align=True)
        layout_row = layout_col.row(align=True)
        layout_split = layout_row.split(factor=0.3, align=True)
        layout_split.label(text="Rigth hand:")
        layout_split.prop(scene, "deformable_mesh_right_ref", icon='MESH_DATA')
        layout_row = layout_col.row(align=True)
        layout_split = layout_row.split(factor=0.3, align=True)
        layout_split.label(text="Left hand:")
        layout_split.prop(scene, "deformable_mesh_left_ref", icon='MESH_DATA')

        layout_shape_col = layout.column(align=True)
        layout_shape_col_path = layout_shape_col.row(align=True)
        layout_shape_path = layout_shape_col_path.split(factor=0.3, align=True)
        layout_shape_path.label(text="Shape Path:")
        layout_shape_path.prop(scene, "shape_path")
        layout_shape_col.operator(VIEW3D_OT_LoadShapes.bl_idname)
        layout_shape_col = layout.column(align=True)
        layout_shape_row = layout_shape_col.row(align=True)
        layout_shape = layout_shape_row.split(factor=0.3, align=True)
        layout_shape.label(text="Shape:")
        layout_shape.prop(scene, "shape_selection")
        layout_apply_shape_row = layout_shape_col.row(align=True)
        layout_apply_shape = layout_apply_shape_row.split(factor=0.7, align=True)
        layout_apply_shape.operator(VIEW3D_OT_ApplyShape.bl_idname)
        layout_apply_shape.operator(VIEW3D_OT_DeleteShape.bl_idname)

        layout_save_col = layout.column(align=True)
        layout_save_col.prop(scene, "shape_name", placeholder="Shape Name")
        layout_sr_row = layout_save_col.row(align=True)
        layout_save_rename = layout_sr_row.split(factor=0.7, align=True)
        layout_save_rename.operator(VIEW3D_OT_SaveShape.bl_idname)
        layout_save_rename.operator(VIEW3D_OT_RenameShape.bl_idname)

        layout_col = layout.column(align=True)
        layout_col.prop(scene, "std_slider")
        layout_row = layout_col.row(align=True)
        layout_split = layout_row.split(factor=0.7, align=True)
        layout_split.operator(VIEW3D_OT_RandomMeshShape.bl_idname)
        layout_split.operator(VIEW3D_OT_ResetMeshShape.bl_idname)
        layout.operator(VIEW3D_OT_UpdateJointPositions.bl_idname)
        layout.operator(VIEW3D_OT_ShapeKeyframe.bl_idname)

class VIEW3D_PT_Sensor(bpy.types.Panel):
    """"""
    bl_label = "Sensor"
    bl_idname = "VIEW3D_PT_Sensor"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Hand Poser"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        # layout.label(text="Sensor:")
        # box_sensor = layout.box()
        layout_add_col = layout.column(align=True)
        layout_add_row = layout_add_col.row(align=True)
        layout_add = layout_add_row.split(factor=0.35, align=True)
        layout_add.label(text="Type:")
        layout_add.prop(scene, "sensor_type")
        layout_add_col.operator(VIEW3D_OT_AddSensor.bl_idname, icon='ADD')
        layout.operator(VIEW3D_OT_MoveSensorToOrigin.bl_idname)
        # box_sensor.separator()
        layout_rand_col = layout.column(align=True)
        layout_row = layout_rand_col.row(align=True)
        layout_split = layout_row.split(factor=0.35, align=True)
        layout_split.label(text="Sample mesh:")
        layout_split.prop(scene, "random_positions_ref", icon='MESH_DATA')
        # box_rand_col.separator(factor=0.2, type='SPACE')
        # box_rand_col.prop(scene, "random_positions_ref")
        layout_row = layout_rand_col.row(align=True)
        layout_split = layout_row.split(factor=0.35, align=True)
        layout_split.label(text="Orientation:")
        layout_split.prop(scene, "sensor_orientation", icon='ORIENTATION_NORMAL')
        # box_rand_col.separator(factor=0.2, type='SPACE')
        layout_rand_col.operator(VIEW3D_OT_RandomSensorPosition.bl_idname)
        """layout_rand_angle = layout.column(align=True)
        layout_rand_angle_row = layout_rand_angle.row(align=True)
        layout_rand_angle_split = layout_rand_angle_row.split(factor=0.1, align=True)
        layout_rand_angle_row.label(text="Number of Rotations:")
        layout_rand_angle_row.prop(scene, "angle_restriction")
        layout_rand_angle.operator(VIEW3D_OT_RandomSensorRotation.bl_idname)"""
        layout.operator(VIEW3D_OT_SensorKeyframe.bl_idname)
        # TODO: add radio button to consider IR from "natural" sources (sun)

class VIEW3D_PT_Dataset(bpy.types.Panel):
    """"""
    bl_label = "Generate Dataset"
    bl_idname = "VIEW3D_PT_Dataset"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Hand Poser"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        
        # layout_row = layout.row(align=True)
        # split_render = layout_row.split(factor=0.9, align=True)
        layout_col = layout.column(align=True)
        layout_col.label(text="Armatures which poses will be keyframed:")
        for i, item in enumerate(scene.pose_arm):
            box = layout_col.box()
            box_row = box.row(align=False)
            box_col = box_row.column(align=True)
            box_row_ord = box_col.row(align=True)
            box_ord = box_row_ord.split(factor=0.27, align=True)
            box_ord.label(text="Group:")
            box_ord.prop(item, "group")
            box_row_arm = box_col.row(align=True)
            box_arm = box_row_arm.split(factor=0.27, align=True)
            box_arm.label(text="Armature:")
            box_arm.prop(item, "arm_ref", icon='ARMATURE_DATA')
            box_row_bone = box_col.row(align=True)
            box_bone = box_row_bone.split(factor=0.27, align=True)
            box_bone.label(text="Bones:")
            box_bone.prop(item, "bone_col", icon='GROUP_BONE')
            col_x = box_row.column(align=True)
            remove_op = col_x.operator("view3d.remove_pose_item", icon='X')
            remove_op.index = i
        layout_col.operator("view3d.add_pose_armature", icon='ADD')
        layout.operator(VIEW3D_OT_GenerateFrames.bl_idname, icon="SEQUENCE")

        layout.separator(type="LINE")
        layout.operator(VIEW3D_OT_MultiviewRender.bl_idname, icon="RENDER_ANIMATION")
        # split_render.operator(VIEW3D_OT_InfoBox.bl_idname, icon="QUESTION")
        layout_row = layout.row(align=True)
        split_meta = layout_row.split(factor=0.6, align=True)
        split_meta.operator(VIEW3D_OT_ExportMetadata.bl_idname, icon="EXPORT")
        split_meta.prop(scene, "file_extension_selection")

classes = (
    VIEW3D_OT_GenerateFrames,
    VIEW3D_OT_MultiviewRender,
    # VIEW3D_OT_InfoBox,
    VIEW3D_OT_ExportMetadata,
    ExportArmatureGroup,
    VIEW3D_OT_AddExportArmature,
    VIEW3D_OT_RemoveExportArmature,
    PoseArmatureGroup,
    VIEW3D_OT_AddPoseArmature,
    VIEW3D_OT_RemovePoseArmature,
    VIEW3D_OT_ImportMANO,
    VIEW3D_OT_AddMANOHand,
    VIEW3D_OT_AttachObject,
    VIEW3D_OT_DetachObject,
    VIEW3D_OT_GeneratePose,
    VIEW3D_OT_ResetPose,
    VIEW3D_OT_LoadPoses,
    VIEW3D_OT_SavePose,
    VIEW3D_OT_RenamePose,
    VIEW3D_OT_ApplyPose,
    VIEW3D_OT_DeletePose,
    VIEW3D_OT_LoadShapes,
    VIEW3D_OT_SaveShape,
    VIEW3D_OT_RenameShape,
    VIEW3D_OT_ApplyShape,
    VIEW3D_OT_DeleteShape,
    VIEW3D_OT_PoseShapes,
    VIEW3D_OT_ArmatureKeyframe,
    VIEW3D_OT_RandomMeshShape,
    VIEW3D_OT_ResetMeshShape,
    VIEW3D_OT_UpdateJointPositions,
    VIEW3D_OT_ShapeKeyframe,
    VIEW3D_OT_AddSensor,
    VIEW3D_OT_ConfigureCompositing,
    VIEW3D_OT_MoveSensorToOrigin,
    VIEW3D_OT_RandomSensorPosition,
    # VIEW3D_OT_RandomSensorRotation,
    VIEW3D_OT_SensorKeyframe,
    VIEW3D_PT_Dataset,
    VIEW3D_PT_Export,
    VIEW3D_PT_MANO_Model,
    VIEW3D_PT_Pose,
    VIEW3D_PT_Shape,
    VIEW3D_PT_Sensor,
)

def register():
    global _cached_poses
    global _cached_pose_attachments
    _cached_poses = [("NONE", "None", "")]
    _cached_pose_attachments = [("NONE", "None", "")]
    reload_modules()
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.export_arm = bpy.props.CollectionProperty(type=ExportArmatureGroup)
    bpy.types.Scene.pose_arm = bpy.props.CollectionProperty(type=PoseArmatureGroup)
    print("Registered")

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    print("Unregistered")

if __name__ == "__main__":
    register()