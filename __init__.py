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
import math
import json
# import os
# import re
import sys
import typing
from math import radians

def ensure_site_packages(packages: typing.List[typing.Tuple[str, str]]):
    """ `packages`: list of tuples (<import name>, <pip name>) """
    
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
    ("flatbuffers", "flatbuffers"),
])

from .VIRTOSHA.FlatBuffers import FrameBatch

def new_sensor_camera(cam_name="Camera"):
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

def new_ir_light(light_name='Spot'):
    spot_data = bpy.data.lights.new(name=light_name, type='SPOT')
    spot_data.energy = 0.1
    spot_data.spot_size = radians(180.0)
    spot_data.spot_blend = 0.3
    return spot_data

def get_sensors_data(sensor_collection):
    sensors_data = []
    for obj in sensor_collection.objects:
        if obj.type == 'EMPTY':
            sens_info = {
                "name": obj.name,
                "location": list(obj.location),
                "rotation_quaternion": list(obj.rotation_quaternion),
            }
            sensors_data.append(sens_info)
    return sensors_data

def get_bones_data(context, obj):
    bones_data = []
    context.view_layer.objects.active = obj
    use_world_space=True
    current_mode = context.mode
    bpy.ops.object.mode_set(mode='POSE')
    for bone in obj.pose.bones:
        if bone.parent == None:
            bone_info = get_bones_list(bone, use_world_space, obj.matrix_world)
            bones_data.append(bone_info)
    bpy.ops.object.mode_set(mode=current_mode)
    return bones_data

def set_compositing_nodetree():
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    tree.nodes.clear() # TODO: let the user deside
    render_layers = tree.nodes.new(type='CompositorNodeRLayers')
    lens_distortion = tree.nodes.new(type='CompositorNodeLensdist')
    hue_correct = tree.nodes.new(type='CompositorNodeHueCorrect')
    hue_correct.name = "BlackAndWhiteFilter"
    composite = tree.nodes.new(type='CompositorNodeComposite')
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
    if bpy.context.scene.light_selection == 'NL':
        hue_correct.mute = True

def setup_scene_later():
    scene = bpy.context.scene
    if scene:
        set_compositing_nodetree()
        return None  # Stop timer
    return 0.5  # Try again in 0.5 seconds

def generate_random_pose(armature):
    bpy.context.view_layer.objects.active = armature
    current_mode = bpy.context.mode
    bpy.ops.object.mode_set(mode='POSE')
    for bone in armature.pose.bones:
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

"""def add_constraints_to_armature(armature_name="Armature"):
    obj = bpy.data.objects.get(armature_name)
    if not obj or obj.type != 'ARMATURE':
        print(f"Armature '{armature_name}' not found.")
        return

    bpy.context.view_layer.objects.active = obj
    current_mode = bpy.context.mode
    bpy.ops.object.mode_set(mode='POSE')

    for bone in obj.pose.bones:
        if bone.name != "Wrist" and not re.match("^Metacarpus", bone.name):
            # Prevent duplicate constraints
            # if not any(c.type == 'LIMIT_ROTATION' for c in bone.constraints):
            constraint = bone.constraints.new('LIMIT_ROTATION')
            constraint.min_z = -1.4
            constraint.max_z = 1.4
            constraint.owner_space = 'LOCAL'
            constraint.use_limit_z = True
            constraint.use_transform_limit = True

    bpy.ops.object.mode_set(mode=current_mode)"""

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

def get_bones_list(bone, use_world_space, matrix_world):
    # Get head/tail in desired space
    if use_world_space:
        head = matrix_world @ bone.head
        tail = matrix_world @ bone.tail
    else:
        head = bone.head
        tail = bone.tail
    bone_info = {}
    bone_info["name"] = bone.name
    bone_info["head"] = list(head)
    bone_info["tail"] = list(tail)
    bone_info["rotation_quaternion"] = list(bone.rotation_quaternion)
    children_info = []
    for child in bone.children:
        child_info = get_bones_list(child, use_world_space, matrix_world)
        children_info.append(child_info)
    bone_info["children"] = children_info
    return bone_info

"""bpy.types.Scene.viewport_checkbox = bpy.props.BoolProperty(
    name="Update in Viewport",
    description="Display render type in viewport",
    default=True,
    update=update_light_selection
)"""

bpy.types.Scene.export_checkbox = bpy.props.BoolProperty(
    name="Export Metadata",
    description="Export sensors and joints positions alongside renders",
    default=True,
#    update=
)

bpy.types.Scene.light_selection = bpy.props.EnumProperty(
    name="Appearance",
    description="Choose an option from the dropdown",
    items=[
        ('NL', "Natural light", "Standard render"),
        ('IR', "IR", "Stylized sensor render")
    ],
    default='NL',
    update=update_light_selection
)


bpy.types.Scene.save_folder = bpy.props.StringProperty(
    name="Save To",
    description="Folder to save rendered images",
    subtype='DIR_PATH'
)

bpy.types.Scene.random_pose_checkbox = bpy.props.BoolProperty(
    name="Use Random Poses",
    description="Generate and render random poses",
    default=False,
#    update=
)

bpy.types.Scene.random_poses_slider = bpy.props.IntProperty(
    name="Number of Poses",
    description="Number of random poses to generate and render",
    default=1,
    min=1,
    max=100
)

bpy.types.Scene.armature_ref = bpy.props.PointerProperty(
    name="Armature",
    type=bpy.types.Object,
    poll=lambda self, obj: obj.type == 'ARMATURE',
)

class VIEW3D_OT_MultiviewRender(bpy.types.Operator):
    """"""
    bl_idname = "view3d.muliview_render"
    bl_label = "Muliview Render"
    bl_description="Render the imgaes from sensors"
    
    def execute(self, context):
        # Check for sensors
        sensor_collection = bpy.data.collections.get('Sensors')
        if not sensor_collection:
            self.report({'ERROR'}, "Sensors collection not found")
            return
        # Check for save folder
        folder = context.scene.save_folder
        if not folder:
            self.report({'ERROR'}, "No save folder selected")
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
        # Save Metadata
        if context.scene.export_checkbox:
            sensors_data = get_sensors_data(sensor_collection)
            armature = context.scene.armature_ref
            if not armature or armature.type != 'ARMATURE':
                self.report({'ERROR'}, f"Please select the armature.")
                return
            bones_data = {}
            # Get metadata
            current_frame = context.scene.frame_current
            for i in range(context.scene.frame_start, context.scene.frame_end+1):
                context.scene.frame_set(i)
                bones_data[f"Frame {i}"] = get_bones_data(context, armature)
                # context.scene.render.filepath = bpy.path.abspath(folder) + "sensor_" + bpy.context.scene.light_selection + f"_Pose_{i+1}"
                # bpy.ops.render.render(write_still=True)
            context.scene.frame_set(current_frame)
            # Save metadata
            export_filepath = bpy.path.abspath(folder) + "metadata.json"
            with open(export_filepath, 'w') as f:
                json.dump({"Sensors":sensors_data, "Bones":bones_data}, f, indent=4)
        # Invoke render
        bpy.ops.render.render(animation=True)
        self.report({'INFO'}, "Render successfully saved")
        return {'FINISHED'}

class VIEW3D_OT_AddSensor(bpy.types.Operator):
    """"""
    bl_idname = "view3d.add_sensor"
    bl_label = "Add Sensor"
    bl_description="Add a sensor to the scene"
    
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
            bpy.context.scene.collection.children.link(target_collection)
        
        # Create an empty
        empty = bpy.data.objects.new(name=empty_name, object_data=None)
        empty.empty_display_type = 'PLAIN_AXES'
        empty.scale = (0.2, 0.2, 0.2)
        target_collection.objects.link(empty)

        # Create cameras
        
        render = bpy.context.scene.render
        view_names = {v.name for v in render.views}
        
        cam_left_data = new_sensor_camera()
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
        
        cam_right_data = new_sensor_camera()
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
        
        spot_left_data = new_ir_light()
        
        spot_obj_left = bpy.data.objects.new(name=f"IR_LED_{index:03}_Left", object_data=spot_left_data)
        spot_obj_left.parent = empty
        spot_obj_left.location = (-0.06, 0.0, 0.0)
        spot_obj_left.scale = (0.1, 0.1, 0.1)
        target_collection.objects.link(spot_obj_left)
        
        spot_right_data = new_ir_light()
        
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
        
        return {'FINISHED'}

class VIEW3D_OT_GeneratePose(bpy.types.Operator):
    """"""
    bl_idname = "view3d.generate_pose"
    bl_label = "Random Pose"
    bl_description="Generate a rondom pose"
    def execute(self, context):
        armature = context.scene.armature_ref
        if not armature:
            self.report({'ERROR'}, f"Please select the armature.")
            return {'CANCELLED'}
        generate_random_pose(armature)
        return {'FINISHED'}
    
class VIEW3D_OT_SetKeyframe(bpy.types.Operator):
    """"""
    bl_idname = "view3d.set_keyframe"
    bl_label = "Keyframe"
    bl_description="Set current pose as a keyframe"
    def execute(self, context):
        armature = context.scene.armature_ref
        if not armature:
            self.report({'ERROR'}, f"Please select the armature.")
            return {'CANCELLED'}
        current_mode = bpy.context.mode
        bpy.ops.object.mode_set(mode='POSE')
        for bone in armature.pose.bones:
            bone.keyframe_insert(data_path="location")
            bone.keyframe_insert(data_path="rotation_euler")
        bpy.ops.object.mode_set(mode=current_mode)
        return {'FINISHED'}

class VIEW3D_OT_ResetPose(bpy.types.Operator):
    """"""
    bl_idname = "view3d.reset_pose"
    bl_label = "Reset Pose"
    bl_description="Reset the pose"
    def execute(self, context):
        armature = context.scene.armature_ref
        if not armature:
            self.report({'ERROR'}, f"Please select the armature.")
            return {'CANCELLED'}
        context.view_layer.objects.active = armature
        current_mode = context.mode
        bpy.ops.object.mode_set(mode='POSE')
        for bone in armature.pose.bones:
            bone.rotation_mode = 'XYZ'
            bone.rotation_euler = (0,0,0)
        bpy.ops.object.mode_set(mode=current_mode)
        return {'FINISHED'}

class VIEW3D_PT_MyCustomPanel(bpy.types.Panel):
    """My Custom Panel in N-Panel"""
    bl_label = ""
    bl_idname = "VIEW3D_PT_my_custom_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Multi-IR Render'

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        
        layout.label(text="Render:")
        box_render = layout.box()
        box_render.prop(scene, "light_selection")
        # box_render.prop(scene, "viewport_checkbox")
        box_render.operator(VIEW3D_OT_MultiviewRender.bl_idname)
        # box_render.prop(scene, "random_pose_checkbox")
        # split_rand = box_render.row()
        # split_rand.enabled = scene.random_pose_checkbox
        # split_rand.prop(scene, "random_poses_slider", slider=True)
        box_render.prop(scene, "export_checkbox")
        box_render.prop(scene, "save_folder")
        
        layout.label(text="Action:")
        box_action = layout.box()
        box_action.prop(context.scene, "armature_ref")
        box_action.operator(VIEW3D_OT_AddSensor.bl_idname)
        box_row = box_action.row(align=True)
        split_pose = box_row.split(factor=0.6, align=True)
        split_pose.operator(VIEW3D_OT_GeneratePose.bl_idname)
        split_pose.operator(VIEW3D_OT_SetKeyframe.bl_idname)
        box_action.operator(VIEW3D_OT_ResetPose.bl_idname)
        # TODO: metall frame as a separate .blend file
        # TODO: sensor model as a separate .blend file
        # TODO: add "Add hand Left/Right" button
        # TODO: set reference point for the sensors
        # TODO: add button to add Natural IR sources

classes = (
    VIEW3D_PT_MyCustomPanel,
    VIEW3D_OT_MultiviewRender,
    VIEW3D_OT_AddSensor,
    VIEW3D_OT_GeneratePose,
    VIEW3D_OT_SetKeyframe,
    VIEW3D_OT_ResetPose,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    # Setup compositing Nodetree
    bpy.app.timers.register(setup_scene_later)
    print("IR Style Render Registered (N-Panel)")

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    print("IR Style Render Unregistered (N-Panel)")

if __name__ == "__main__":
    register()