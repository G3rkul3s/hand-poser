bl_info = {
    "name": "IR Style Render",
    "author": "Nikita Morev",
    "version": (0, 1),
    "blender": (2, 80, 0),
    "location": "3D Viewport > Sidebar (N-Panel) > Multi-IR Render",
    "description": "Adds a custom panel to the 3D Viewport's N-Panel for IR simulated sensor render.",
    "warning": "",
    "doc_url": "",
    "category": "Render",
}

import bpy
import random
import numpy as np
import math
import json
# import os
# import re
# import sys
# import typing
from math import radians
from mathutils import Vector, Quaternion, Matrix

if "bpy" in locals():
    import importlib
    if "load_mano" in locals():
        importlib.reload(load_mano)
from .load_mano import load_mano_hand, load_regressor, MANO_BONE_NAMES, FINGERTIPS

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

def update_light_selection(self, context):
    """
    This function is called whenever 'light_selection' changes.
    """
    sensor_collection = bpy.data.collections.get('Sensors')
    if not sensor_collection:
        print("Sensors collection not found")
        return
    sensor_objects = set(sensor_collection.objects)
    selection = self.light_selection
    
    nl_render = True if selection == 'NL' else False
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
    
    hue_correct = context.scene.node_tree.nodes.get("BlackAndWhiteFilter")
    if hue_correct:
        hue_correct.mute = nl_render

def update_joint_positions(armature_obj, J_regressor, vert_shaped, context):
    """
    Updates the joint (bone) positions in the armature based on the current shape of the mesh.
    """
    # Compute new joint positions
    joints = J_regressor @ vert_shaped # shape (16, 3)
    active_curr = context.view_layer.objects.active
    context.view_layer.objects.active = armature_obj
    current_mode = context.mode
    bpy.ops.object.mode_set(mode='EDIT')
    edit_bones = armature_obj.data.edit_bones

    for i, name in enumerate(MANO_BONE_NAMES):
        if name not in edit_bones:
            continue
        bone = edit_bones[name]
        new_head = joints[i]
        new_tail = bone.tail + Vector(new_head - bone.head)
        bone.head = new_head
        bone.tail = new_tail
    
    for name, index in FINGERTIPS.items():
        bone = edit_bones[name]
        new_head = vert_shaped[index]
        new_tail = bone.tail + Vector(new_head - bone.head)
        bone.head = new_head
        bone.tail = new_tail

    bpy.ops.object.mode_set(mode=current_mode)
    context.view_layer.objects.active = active_curr

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
    description="Choose a lighting for the scene",
    items=[
        ('NL', "Natural", "Use natural lighting"),
        ('IR', "Infrared", "Use infrared lighting")
    ],
    default='NL',
    update=update_light_selection
)

bpy.types.Scene.hand_selection = bpy.props.EnumProperty(
    name="Hand",
    description="Choose a MANO hand type",
    items=[
        ('LEFT', "Left", "Left hand"),
        ('RIGHT', "Right", "Right hand")
    ],
    default='RIGHT',
)

bpy.types.Scene.file_extension_selection = bpy.props.EnumProperty(
    name="",
    description="Choose a file extension",
    items=[
        ('JSON', ".json", "Export to .json"),
        # ('FBS', ".fbs", "Export to .fbs")
    ],
    default='JSON',
)

bpy.types.Scene.origin_ref = bpy.props.PointerProperty(
    name="",
    description = "Select a reference frame for the sensors and joints positions. \nLeave empty to use world coordinates",
    type=bpy.types.Object,
)

bpy.types.Scene.save_folder = bpy.props.StringProperty(
    name="",
    description="Path to the folder for rendered images and metadata",
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
    name="Armature",
    type=bpy.types.Object,
    poll=lambda self, obj: obj.type == 'ARMATURE',
)

bpy.types.Scene.deformable_mesh_right_ref = bpy.props.PointerProperty(
    name="",
    description = "Right hand mesh with defined shape keys (the same keys as for the left hand)",
    type=bpy.types.Object,
    poll=lambda self, obj: obj.type == 'MESH' and obj.data.shape_keys,
)

bpy.types.Scene.deformable_mesh_left_ref = bpy.props.PointerProperty(
    name="",
    description = "Left hand mesh with defined shape keys (the same keys as for the right hand)",
    type=bpy.types.Object,
    poll=lambda self, obj: obj.type == 'MESH' and obj.data.shape_keys,
)

bpy.types.Scene.random_positions_ref = bpy.props.PointerProperty(
    name="",
    description = "A mesh to randomly sample it's vertecies for the sensor placement",
    type=bpy.types.Object,
    poll=lambda self, obj: obj.type == 'MESH',
)

def get_bone_collections(self, context):
    items = [("NONE", "None", "Use the whole armature")]
    armature = context.scene.armature_ref
    if armature and armature.type == 'ARMATURE':
        items.extend([(bc.name, bc.name, "") for bc in armature.data.collections])
    return items

bpy.types.Scene.selected_bone_collection = bpy.props.EnumProperty(
    name="",
    items=get_bone_collections,
    # default="NONE",
)

bpy.types.Scene.sensor_orientation = bpy.props.EnumProperty(
    name="",
    description="Choose a sensor orientation (where it points)",
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

class VIEW3D_OT_MultiviewRender(bpy.types.Operator):
    """"""
    bl_idname = "view3d.muliview_render"
    bl_label = "Render Animation"
    bl_description="Render the imgaes from sensors for all frames"
    
    def execute(self, context):
        # Check for sensors
        sensor_collection = bpy.data.collections.get('Sensors')
        if not sensor_collection:
            self.report({'ERROR'}, "Sensors collection not found")
            return
        # Check for save folder
        folder = context.scene.save_folder
        if not folder:
            self.report({'ERROR'}, "Save folder not selected")
            return {'CANCELLED'}
        # Set up multiview render
        context.scene.render.use_multiview = True
        context.scene.render.filepath = bpy.path.abspath(folder + "Frame_")
        sensor_names = {obj.name for obj in sensor_collection.objects if obj.type == 'CAMERA'}
        views = context.scene.render.views
        for v in list(views):
            if v.name == 'left' or v.name == 'right':
                v.use = False
            elif v.name not in sensor_names:
#                views.remove(v)
                v.use = False
            # else:
            #     v.use = True
        # Invoke render
        bpy.ops.render.render(animation=True)
        self.report({'INFO'}, "Render successfully saved")
        return {'FINISHED'}

class VIEW3D_OT_AddSensor(bpy.types.Operator):
    """"""
    bl_idname = "view3d.add_sensor"
    bl_label = "Add"
    bl_description="Add a sensor to the scene"
    bl_options = {'REGISTER', 'UNDO'}
    
    def new_sensor_camera(self, cam_name="Camera"):
        cam_data = bpy.data.cameras.new(name=cam_name)
        # cam_data.type = 'PANO'
        # cam_data.panorama_type = 'FISHEYE_EQUISOLID'
        # cam_data.fisheye_lens = 1.50
        # cam_data.fisheye_fov = radians(170.0)
        cam_data.type = 'PERSP'
        cam_data.lens = 2.1
        cam_data.clip_start = 0.001
        cam_data.clip_end = 50
        cam_data.sensor_fit = 'AUTO'
        # cam_data.sensor_width = 4
        cam_data.sensor_width = 14
        cam_data.display_size = 0.3
        return cam_data

    def new_ir_light(self, light_name='Spot'):
        spot_data = bpy.data.lights.new(name=light_name, type='SPOT')
        spot_data.energy = 0.1
        spot_data.spot_size = radians(180.0)
        spot_data.spot_blend = 0.3
        return spot_data

    def execute(self, context):
        collection_name  = "Sensors"
        base_name = "Sensor"
        index = 1
        while f"{base_name}.{index:03}" in bpy.data.objects:
            index += 1
        empty_name = f"{base_name}.{index:03}"
        
        target_collection = bpy.data.collections.get(collection_name)
        if not target_collection:
            target_collection = bpy.data.collections.new(collection_name)
            context.scene.collection.children.link(target_collection)
        
        # Create an empty
        empty = bpy.data.objects.new(name=empty_name, object_data=None)
        empty.empty_display_type = 'PLAIN_AXES'
        empty.scale = (0.2, 0.2, 0.2)
        target_collection.objects.link(empty)

        # Create cameras
        
        render = context.scene.render
        view_names = {v.name for v in render.views}
        
        cam_left_data = self.new_sensor_camera()
        cam_left_obj = bpy.data.objects.new(name=f"Camera_Sensor_{index:03}_Left", 
                                            object_data=cam_left_data)
        cam_left_obj.parent = empty
        cam_left_obj.location = (-0.032, 0.0, 0.0)
        cam_left_obj.scale = (0.2, 0.2, 0.2)
        target_collection.objects.link(cam_left_obj)
        
        cam_left_name = f"Camera_Sensor_{index:03}_Left"
        if cam_left_name not in view_names:
            cam_left_rv = render.views.new(name=cam_left_name)
            cam_left_rv.camera_suffix = f"_Sensor_{index:03}_Left"
            cam_left_rv.use = True
            # cam_left_rv.file_suffix = ""
        
        cam_right_data = self.new_sensor_camera()
        cam_right_obj = bpy.data.objects.new(name=f"Camera_Sensor_{index:03}_Right", 
                                            object_data=cam_right_data)
        cam_right_obj.parent = empty
        cam_right_obj.location = (0.032, 0.0, 0.0)
        cam_right_obj.scale = (0.2, 0.2, 0.2)
        target_collection.objects.link(cam_right_obj)
        
        cam_right_name = f"Camera_Sensor_{index:03}_Right"
        if cam_right_name not in view_names:
            cam_right_rv = render.views.new(name=cam_right_name)
            cam_right_rv.camera_suffix = f"_Sensor_{index:03}_Right"
            cam_right_rv.use = True
            # cam_right_rv.file_suffix = ""
        
        # Create IR LEDs
        
        spot_left_data = self.new_ir_light()
        
        spot_obj_left = bpy.data.objects.new(name=f"IR_LED_{index:03}_Left", object_data=spot_left_data)
        spot_obj_left.parent = empty
        spot_obj_left.location = (-0.06, 0.0, 0.0)
        spot_obj_left.scale = (0.1, 0.1, 0.1)
        target_collection.objects.link(spot_obj_left)
        
        spot_right_data = self.new_ir_light()
        
        spot_obj_right = bpy.data.objects.new(name=f"IR_LED_{index:03}_Right", object_data=spot_right_data)
        spot_obj_right.parent = empty
        spot_obj_right.location = (0.06, 0.0, 0.0)
        spot_obj_right.scale = (0.1, 0.1, 0.1)
        target_collection.objects.link(spot_obj_right)

        # Set visibility
        nl_render = True if context.scene.light_selection == 'NL' else False
        spot_obj_left.hide_render = nl_render
        spot_obj_right.hide_render = nl_render
        # if context.scene.viewport_checkbox:
        spot_obj_left.hide_viewport = nl_render
        spot_obj_right.hide_viewport = nl_render
        
        context.view_layer.objects.active = empty
        empty.select_set(True)
        return {'FINISHED'}

class VIEW3D_OT_GeneratePose(bpy.types.Operator):
    """"""
    bl_idname = "view3d.generate_pose"
    bl_label = "Random Pose"
    bl_description="Generate a rondom pose"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            return (context.scene.armature_ref)
        except: return False

    def execute(self, context):
        armature = context.scene.armature_ref
        active_curr = context.view_layer.objects.active
        context.view_layer.objects.active = armature
        current_mode = context.mode
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
        return {'FINISHED'}
    
class VIEW3D_OT_ArmatureKeyframe(bpy.types.Operator):
    """"""
    bl_idname = "view3d.armature_keyframe"
    bl_label = "Keyframe"
    bl_description = "Set the current pose as a keyframe"
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
        context.view_layer.objects.active = armature
        current_mode = bpy.context.mode
        bpy.ops.object.mode_set(mode='POSE')
        for bone in armature.pose.bones:
            if bone_collection and bone.name not in bone_collection.bones:
                continue
            bone.keyframe_insert(data_path="location")
            bone.keyframe_insert(data_path="rotation_quaternion")
        bpy.ops.object.mode_set(mode=current_mode)
        context.view_layer.objects.active = active_curr
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
        context.view_layer.objects.active = armature
        current_mode = context.mode
        bpy.ops.object.mode_set(mode='POSE')
        for bone in armature.pose.bones:
            if bone_collection and bone.name not in bone_collection.bones:
                continue
            bone.rotation_mode = 'XYZ'
            bone.rotation_euler = (0,0,0)
        bpy.ops.object.mode_set(mode=current_mode)
        context.view_layer.objects.active = active_curr
        return {'FINISHED'}

class VIEW3D_OT_ExportMetadata(bpy.types.Operator):
    """"""
    bl_idname = "view3d.export_metadata"
    bl_label = "Export Metadata"
    bl_description="Export metadata for each animation frame"

    def get_sensors_data(self, context, sensor_collection, matrix_world):
        sensors_data = []
        for obj in sensor_collection.objects:
            if obj.type == 'EMPTY':
                location = obj.location
                if context.scene.origin_ref:
                    location = matrix_world @ obj.location
                sens_info = {
                    "name": obj.name,
                    "location": list(location),
                    "rotation_quaternion": list(obj.rotation_quaternion),
                }
                sensors_data.append(sens_info)
        return sensors_data

    def get_bone_data(self, bone, matrix_world, armature_world):
        head = matrix_world @ (armature_world @ bone.head)
        bone_info = {}
        bone_info["name"] = bone.name
        bone_info["location"] = list(head)
        bone_info["rotation_quaternion"] = list(bone.rotation_quaternion)
        return bone_info

    def get_armature_data(self, context, armature_collection, matrix_world):
        armature_data = []
        active_curr = context.view_layer.objects.active
        current_mode = context.mode
        for armature in armature_collection:
            context.view_layer.objects.active = armature
            bpy.ops.object.mode_set(mode='POSE')
            armature_info = {}
            armature_info["name"] = armature.name
            armature_info["joints"] = []
            context.view_layer.objects.active = armature
            for bone in armature.pose.bones:
                bone_info = self.get_bone_data(bone, matrix_world, armature.matrix_world)
                armature_info["joints"].append(bone_info)
            armature_data.append(armature_info)
        bpy.ops.object.mode_set(mode=current_mode)
        context.view_layer.objects.active = active_curr
        return armature_data

    def execute(self, context):
        # Check for save folder
        folder = context.scene.save_folder
        if not folder:
            self.report({'ERROR'}, "No save folder selected")
            return {'CANCELLED'}
        sensor_collection = bpy.data.collections.get('Sensors')
        armature_collection = [obj for obj in bpy.data.collections.get('Export armature').objects if obj.type == 'ARMATURE']
        # Get the reference frame
        origin = context.scene.origin_ref
        if origin:
            rot = origin.rotation_euler.to_matrix().to_4x4()
            trans = Matrix.Translation(origin.location)
            matrix_world = (trans @ rot).inverted()
        else:
            matrix_world = Matrix.Translation(Vector((0,0,0)))
        print(matrix_world)
        # Iterate over all frames
        current_frame = context.scene.frame_current
        for i in range(context.scene.frame_start, context.scene.frame_end+1):
            context.scene.frame_set(i)
            # Get sensor(s) metadata
            sensors_data = self.get_sensors_data(context, sensor_collection, matrix_world) if sensor_collection else []
            # Get armature(s) metadata
            armature_data = self.get_armature_data(context, armature_collection, matrix_world) if armature_collection else []
            # Save metadata
            export_filepath = bpy.path.abspath(folder) + f"Frame_{i:03}_metadata.json"
            with open(export_filepath, 'w') as f:
                json.dump({"Sensor data":sensors_data, "Armature data":armature_data}, f, indent=4)
        context.scene.frame_set(current_frame)
        self.report({'INFO'}, "Metadata successfully exported")
        return {'FINISHED'}

class VIEW3D_OT_InfoBox(bpy.types.Operator):
    bl_idname = "view3d.info_box"
    bl_label = ""
    bl_description = "It is recomended to use a rendering script.\n" \
    "Rendering the animation inside the Blender GUI will freeze the application"
    
    def execute(self, context):
        self.report({'INFO'}, "It is recomended to use a rendering script. " \
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
    bl_description = "Set random sensor position on the sampling mesh"
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
            for key in mesh_right.data.shape_keys.key_blocks:
                key.value = 0.0
        mesh_left = context.scene.deformable_mesh_left_ref
        if mesh_left:
            for key in mesh_left.data.shape_keys.key_blocks:
                key.value = 0.0
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
            if len(mesh_right.data.shape_keys.key_blocks) != len(mesh_left.data.shape_keys.key_blocks):
                self.report({'ERROR'}, "Meshes have different number of shape keys")
            for key_right, key_left in zip(mesh_right.data.shape_keys.key_blocks, mesh_left.data.shape_keys.key_blocks):
                key_right.value = key_left.value = random.gauss(0.0, context.scene.std_slider)
        elif mesh_right:
            for key_right in mesh_right.data.shape_keys.key_blocks:
                key_right.value = random.gauss(0.0, context.scene.std_slider)
        elif mesh_left:
            for key_left in mesh_left.data.shape_keys.key_blocks:
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
            for key in mesh_right.data.shape_keys.key_blocks:
                key.keyframe_insert(data_path="value")
        mesh_left = context.scene.deformable_mesh_left_ref
        if mesh_left:
            for key in mesh_left.data.shape_keys.key_blocks:
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
            vertices = np.array([v.co[:] for v in eval_mesh.vertices])
            # Re-enable the Armature modifier
            for i, mod in enumerate(mesh_right.modifiers):
                if mod.type == 'ARMATURE':
                    mod.show_viewport = mod_show_list[i]
            # Cash the regressor and the template
            if self.J_regressor_right is None:
                self.J_regressor_right = load_regressor('RIGHT')
            update_joint_positions(mesh_right.parent, self.J_regressor_right, vertices, context)
        
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
            vertices = np.array([v.co[:] for v in eval_mesh.vertices])
            # Re-enable the Armature modifier
            for i, mod in enumerate(mesh_left.modifiers):
                if mod.type == 'ARMATURE':
                    mod.show_viewport = mod_show_list[i]
            # Cash the regressor and the template
            if self.J_regressor_left is None:
                self.J_regressor_left = load_regressor('LEFT')
            update_joint_positions(mesh_left.parent, self.J_regressor_left, vertices, context)
        return{'FINISHED'}
    
class VIEW3D_OT_AddMANOHand(bpy.types.Operator):
    bl_idname = "view3d.add_mano_hand"
    bl_label = "Add"
    bl_description = "Add a MANO hand model"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = load_mano_hand(context.scene.hand_selection)
        obj.location = context.scene.cursor.location
        target_collection = bpy.data.collections.get("Export armature")
        if not target_collection:
            target_collection = bpy.data.collections.new("Export armature")
            context.scene.collection.children.link(target_collection)
        if obj.name not in target_collection.objects:
            target_collection.objects.link(obj) # NOTE: this creates a duplicate
        return{'FINISHED'}

class VIEW3D_OT_ConfigureCompositing(bpy.types.Operator):
    bl_idname = "view3d.configure_compositing"
    bl_label = "Configure Compositing"
    bl_description = "Configure the compositing node tree to simulate the infrared sensor"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        context.scene.use_nodes = True
        tree = context.scene.node_tree
        if context.scene.override_compositing:
            tree.nodes.clear()
        render_layers = context.scene.node_tree.nodes.get("Render Layers")
        if not render_layers:
            render_layers = tree.nodes.new(type='CompositorNodeRLayers')
        composite = context.scene.node_tree.nodes.get("Composite")
        if not composite:
            composite = tree.nodes.new(type='CompositorNodeComposite')
        lens_distortion = tree.nodes.new(type='CompositorNodeLensdist')
        hue_correct = tree.nodes.new(type='CompositorNodeHueCorrect')
        hue_correct.name = "BlackAndWhiteFilter"
        render_layers.location = (0, 0)
        lens_distortion.location = (300, 0)
        hue_correct.location = (500, 0)
        composite.location = (900, 0)
        # Set the distortion to 1.0
        lens_distortion.inputs["Distortion"].default_value = 1.0
        # Set the saturation curve to a flat line at 0.0
        sat_curve = hue_correct.mapping.curves[1]  # 0=H, 1=S, 2=V
        # Remove existing points
        for i in reversed(range(2, len(sat_curve.points))):
            sat_curve.points.remove(sat_curve.points[i])
        sat_curve.points[0].location = (0.0, 0.0)
        sat_curve.points[1].location = (1.0, 0.0)
        # Link the nodes together
        tree.links.new(render_layers.outputs['Image'], lens_distortion.inputs['Image'])
        tree.links.new(lens_distortion.outputs['Image'], hue_correct.inputs['Image'])
        tree.links.new(hue_correct.outputs['Image'], composite.inputs['Image'])
        if context.scene.light_selection == 'NL':
            hue_correct.mute = True
        return{'FINISHED'}

class VIEW3D_PT_Export(bpy.types.Panel):
    """"""
    bl_label = "Export / Render"
    bl_idname = "VIEW3D_PT_Export"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Multi-IR Render'

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        
        layout_row = layout.row(align=True)
        split_light = layout_row.split(factor=0.3, align=True)
        split_light.label(text="Lighting:")
        split_light.prop(scene, "light_selection")
        layout_row = layout.row(align=True)
        split_origin = layout_row.split(factor=0.3, align=True)
        split_origin.label(text="World Origin:")
        split_origin.prop(scene, "origin_ref")
        layout_row = layout.row(align=True)
        split_compositing = layout_row.split(factor=0.7, align=True)
        split_compositing.operator(VIEW3D_OT_ConfigureCompositing.bl_idname)
        split_compositing.prop(scene, "override_compositing")
        layout.separator(type='LINE')
        layout_row = layout.row(align=True)
        split_render = layout_row.split(factor=0.9, align=True)
        split_render.operator(VIEW3D_OT_MultiviewRender.bl_idname)
        split_render.operator(VIEW3D_OT_InfoBox.bl_idname, icon="QUESTION")
        layout_row = layout.row(align=True)
        split_meta = layout_row.split(factor=0.6, align=True)
        split_meta.operator(VIEW3D_OT_ExportMetadata.bl_idname)
        split_meta.prop(scene, "file_extension_selection")
        layout_row = layout.row(align=True)
        split_save = layout_row.split(factor=0.3, align=True)
        split_save.label(text="Save Folder:")
        split_save.prop(scene, "save_folder")

class VIEW3D_PT_MANO_Model(bpy.types.Panel):
    """"""
    bl_label = "MANO Hand Model"
    bl_idname = "VIEW3D_PT_MANO_Model"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Multi-IR Render'

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        
        layout.prop(scene, "hand_selection")
        layout.operator(VIEW3D_OT_AddMANOHand.bl_idname)
        # TODO: create separate .blend with SMPL-x with my hands ??

class VIEW3D_PT_Pose(bpy.types.Panel):
    """"""
    bl_label = "Pose"
    bl_idname = "VIEW3D_PT_Pose"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Multi-IR Render'

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        # layout.label(text="Pose:")
        # box_action = layout.box()
        layout.prop(scene, "armature_ref")
        layout_row = layout.row(align=True)
        layout_split = layout_row.split(factor=0.4, align=True)
        layout_split.label(text="Bone Collection:")
        layout_split.prop(scene, "selected_bone_collection")
        layout_row = layout.row(align=True)
        layout_split = layout_row.split(factor=0.7, align=True)
        layout_split.operator(VIEW3D_OT_GeneratePose.bl_idname)
        layout_split.operator(VIEW3D_OT_ResetPose.bl_idname)
        layout.operator(VIEW3D_OT_ArmatureKeyframe.bl_idname)

class VIEW3D_PT_Shape(bpy.types.Panel):
    """"""
    bl_label = "Shape"
    bl_idname = "VIEW3D_PT_Shape"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Multi-IR Render'

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        layout_col = layout.column(align=True)
        layout_row = layout_col.row(align=True)
        layout_split = layout_row.split(factor=0.3, align=True)
        layout_split.label(text="Rigth hand:")
        layout_split.prop(scene, "deformable_mesh_right_ref")
        layout_row = layout_col.row(align=True)
        layout_split = layout_row.split(factor=0.3, align=True)
        layout_split.label(text="Left hand:")
        layout_split.prop(scene, "deformable_mesh_left_ref")
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
    bl_category = "Multi-IR Render"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        # layout.label(text="Sensor:")
        # box_sensor = layout.box()
        layout.operator(VIEW3D_OT_AddSensor.bl_idname)
        layout.operator(VIEW3D_OT_MoveSensorToOrigin.bl_idname)
        # box_sensor.separator()
        layout_rand_col = layout.column(align=True)
        layout_row = layout_rand_col.row(align=True)
        layout_split = layout_row.split(factor=0.35, align=True)
        layout_split.label(text="Sample mesh:")
        layout_split.prop(scene, "random_positions_ref")
        # box_rand_col.separator(factor=0.2, type='SPACE')
        # box_rand_col.prop(scene, "random_positions_ref")
        layout_row = layout_rand_col.row(align=True)
        layout_split = layout_row.split(factor=0.35, align=True)
        layout_split.label(text="Orientation:")
        layout_split.prop(scene, "sensor_orientation")
        # box_rand_col.separator(factor=0.2, type='SPACE')
        layout_rand_col.operator(VIEW3D_OT_RandomSensorPosition.bl_idname)
        """layout_rand_angle = layout.column(align=True)
        layout_rand_angle_row = layout_rand_angle.row(align=True)
        layout_rand_angle_split = layout_rand_angle_row.split(factor=0.1, align=True)
        layout_rand_angle_row.label(text="Number of Rotations:")
        layout_rand_angle_row.prop(scene, "angle_restriction")
        layout_rand_angle.operator(VIEW3D_OT_RandomSensorRotation.bl_idname)"""
        layout.operator(VIEW3D_OT_SensorKeyframe.bl_idname)
        # TODO: add button to consider Natural IR sources
        # TODO: metall frame + sensor as a separate .blend file ???

classes = (
    VIEW3D_OT_MultiviewRender,
    VIEW3D_OT_InfoBox,
    VIEW3D_OT_ExportMetadata,
    VIEW3D_OT_AddMANOHand,
    VIEW3D_OT_GeneratePose,
    VIEW3D_OT_ResetPose,
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
    VIEW3D_PT_Export,
    VIEW3D_PT_MANO_Model,
    VIEW3D_PT_Pose,
    VIEW3D_PT_Shape,
    VIEW3D_PT_Sensor,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    print("IR Style Render Registered (N-Panel)")

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    print("IR Style Render Unregistered (N-Panel)")

if __name__ == "__main__":
    register()