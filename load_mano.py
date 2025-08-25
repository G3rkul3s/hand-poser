import bpy
import numpy as np
# import os
from pathlib import Path
from mathutils import Vector

# === CONFIG ===
BONE_NAMES = [              # 16 MANO bones:
    "Wrist",                # "wrist",
    "IndexProximal",        # "index1",
    "IndexIntermadiate",    # "index2",
    "IndexDistal",          # "index3",
    "MiddleProximal",       # "middle1",
    "MiddleIntermadiate",   # "middle2",
    "MiddleDistal",         # "middle3",
    "LittleProximal",       # "pinky1",
    "LittleIntermadiate",   # "pinky2",
    "LittleDistal",         # "pinky3",
    "RingProximal",         # "ring1",
    "RingIntermadiate",     # "ring2",
    "RingDistal",           # "ring3",
    "ThumbMetacarpal",      # "thumb1",
    "ThumbProximal",        # "thumb2",
    "ThumbDistal",          # "thumb3",
    # NOTE: Append new bones at the end
]
FINGERTIP_NAMES = [
    "ThumbTip",
    "IndexTip",
    "MiddleTip",
    "RingTip",
    "LittleTip",
]
FINGERTIPS = {
    FINGERTIP_NAMES[0] : 744,
    FINGERTIP_NAMES[1] : 320,
    FINGERTIP_NAMES[2] : 443,
    FINGERTIP_NAMES[3] : 554,
    FINGERTIP_NAMES[4] : 671,
}
BONE_PARENTS = {
    BONE_NAMES[0]       : None,
    BONE_NAMES[13]      : BONE_NAMES[0],
    BONE_NAMES[14]      : BONE_NAMES[13],
    BONE_NAMES[15]      : BONE_NAMES[14],
    FINGERTIP_NAMES[0]  : BONE_NAMES[15],
    BONE_NAMES[1]       : BONE_NAMES[0],
    BONE_NAMES[2]       : BONE_NAMES[1],
    BONE_NAMES[3]       : BONE_NAMES[2],
    FINGERTIP_NAMES[1]  : BONE_NAMES[3],
    BONE_NAMES[4]       : BONE_NAMES[0],
    BONE_NAMES[5]       : BONE_NAMES[4],
    BONE_NAMES[6]       : BONE_NAMES[5],
    FINGERTIP_NAMES[2]  : BONE_NAMES[6],
    BONE_NAMES[10]      : BONE_NAMES[0],
    BONE_NAMES[11]      : BONE_NAMES[10],
    BONE_NAMES[12]      : BONE_NAMES[11],
    FINGERTIP_NAMES[3]  : BONE_NAMES[12],
    BONE_NAMES[7]       : BONE_NAMES[0],
    BONE_NAMES[8]       : BONE_NAMES[7],
    BONE_NAMES[9]       : BONE_NAMES[8],
    FINGERTIP_NAMES[4]  : BONE_NAMES[9],
}

def load_mano_model(hand):
    ROOT_DIR = Path(__file__).parent
    mano_path = ROOT_DIR / 'data' / f"MANO_{hand}.npz"
    data = np.load(mano_path)
    v_template = data['v_template']   # [778, 3]
    shapedirs = data['shapedirs']     # [10, 778, 3]
    faces = data['f']                 # [1538, 3]
    joints = data['J']
    weights = data['weights']
    return v_template, shapedirs, faces, joints, weights

def load_regressor(hand):
    ROOT_DIR = Path(__file__).parent
    mano_path = ROOT_DIR / 'data' / f"MANO_{hand}.npz"
    data = np.load(mano_path)
    j_regressor = data['J_regressor']
    return j_regressor

def create_mano_mesh(name, vertices, faces):
    # Create mesh and object
    mesh = bpy.data.meshes.new(name + "_shapes")
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)

    # Create mesh from data
    mesh.from_pydata([tuple(v) for v in vertices], [], [tuple(f) for f in faces])
    mesh.update()

    return obj

def add_shape_keys(obj, shapedirs, base_vertices):
    if obj.data.shape_keys is None:
        obj.shape_key_add(name="Basis")

    for i, shape in enumerate(shapedirs):
        key = obj.shape_key_add(name=f"Shape_{i+1}", from_mix=False)
        key.slider_max = 5.0
        key.slider_min = -5.0
        for v_idx, delta in enumerate(shape):
            key.data[v_idx].co = base_vertices[v_idx] + delta

def create_joint_armature(mesh, hand, joint_positions, bone_names, bone_parents):
    armature_data = bpy.data.armatures.new(f"MANO_{hand}_Hand")
    armature_obj = bpy.data.objects.new(f"MANO_{hand}_Hand", armature_data)
    bpy.context.collection.objects.link(armature_obj)

    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='EDIT')
    bones = {}

    for i, name in enumerate(bone_names):
        bone = armature_data.edit_bones.new(name)
        head = joint_positions[i]
        tail = head + Vector((0.0, 1.0, 0.0)) * 0.04
        bone.head = head
        bone.tail = tail
        bones[name] = bone

    # add joints for fingertips
    for name, index in FINGERTIPS.items():
        bone = armature_data.edit_bones.new(name)
        head = mesh.data.vertices[index].co
        tail = head + Vector((0.0, 1.0, 0.0)) * 0.04
        bone.head = head
        bone.tail = tail
        bones[name] = bone

    # Set up parent relationships
    for name, parent_name in bone_parents.items():
        if parent_name and name in bones and parent_name in bones:
            bones[name].parent = bones[parent_name]

    bpy.ops.object.mode_set(mode='OBJECT')
    return armature_obj

def assign_skinning_weights(obj, weights, bone_names):
    for joint_idx, bone_name in enumerate(bone_names):
        # Create vertex group for each joint
        vg = obj.vertex_groups.new(name=bone_name)
        for v_idx, w in enumerate(weights[:, joint_idx]):
            if w > 0:
                vg.add([v_idx], w, 'REPLACE')

def add_constraints_to_armature(armature):
    bpy.context.view_layer.objects.active = armature
    current_mode = bpy.context.mode
    bpy.ops.object.mode_set(mode='POSE')
    
    root = armature.pose.bones.get(BONE_NAMES[0])

    for bone in root.children:
        if bone.name == BONE_NAMES[13]:
            pass
            # constraint = bone.constraints.new('LIMIT_ROTATION')
            # constraint.min_x = -0.35
            # constraint.max_x = 2.10
            # constraint.use_limit_x = True
            # constraint.min_y = -0.60
            # constraint.max_y = 0.56
            # constraint.use_limit_y = True
            # constraint.min_z = -0.3
            # constraint.max_z = 0.43
            # constraint.use_limit_z = True
            # constraint.owner_space = 'LOCAL'
            # constraint.use_transform_limit = True

            # bone_1 = bone.children[0]
            # constraint = bone_1.constraints.new('LIMIT_ROTATION')
            # constraint.use_limit_x = True
            # constraint.use_limit_y = True
            # constraint.min_z = -0.57
            # constraint.max_z = 1.88
            # constraint.use_limit_z = True
            # constraint.owner_space = 'LOCAL'
            # constraint.use_transform_limit = True

        else:
            constraint = bone.constraints.new('LIMIT_ROTATION')
            constraint.use_limit_x = True
            if bone.name == BONE_NAMES[7]: # if pinky
                constraint.max_y = 0.77
            elif bone.name == BONE_NAMES[10]: # if ring
                constraint.max_y = 0.38
            elif bone.name == BONE_NAMES[4]: # if middle
                constraint.max_y = 0.2
            elif bone.name == BONE_NAMES[1]: # if index
                constraint.min_y = -0.14
            constraint.use_limit_y = True
            constraint.min_z = -0.57  # first layer
            constraint.max_z = 1.74   # first layer
            constraint.use_limit_z = True
            constraint.owner_space = 'LOCAL'
            constraint.use_transform_limit = True

            bone_1 = bone.children[0]
            constraint = bone_1.constraints.new('LIMIT_ROTATION')
            constraint.use_limit_x = True
            constraint.use_limit_y = True
            constraint.min_z = -0.57  # second layer
            constraint.max_z = 1.88   # second layer
            constraint.use_limit_z = True
            constraint.owner_space = 'LOCAL'
            constraint.use_transform_limit = True

            bone_2 = bone_1.children[0]
            constraint = bone_2.constraints.new('LIMIT_ROTATION')
            constraint.use_limit_x = True
            constraint.use_limit_y = True
            constraint.min_z = -0.17  # third layer
            constraint.max_z = 1.57   # third layer
            constraint.use_limit_z = True
            constraint.owner_space = 'LOCAL'
            constraint.use_transform_limit = True

            bone_3 = bone_2.children[0]
            constraint = bone_3.constraints.new('LIMIT_ROTATION')
            constraint.use_limit_x = True
            constraint.use_limit_y = True
            constraint.use_limit_z = True
            constraint.owner_space = 'LOCAL'
            constraint.use_transform_limit = True

    bpy.ops.object.mode_set(mode=current_mode)

def load_mano_hand(hand: str):
    """
    Parameters
    ----------
    hand : str
        'LEFT' or 'RIGHT' hand
    """

    v_template, shapedirs, faces, joints, weights = load_mano_model(hand)

    # === Create base mesh ===
    mesh = create_mano_mesh(f"MANO_{hand}_Hand_mesh", v_template, faces)

    # === Add shape keys ===
    add_shape_keys(mesh, shapedirs, v_template)
    
    # === Add armature ===
    arm = create_joint_armature(mesh, hand, joints, BONE_NAMES, BONE_PARENTS)
    mesh.parent = arm
    
    arm_mod = mesh.modifiers.new(name="ArmatureDeform", type='ARMATURE')
    arm_mod.object = arm
    assign_skinning_weights(mesh, weights, BONE_NAMES)

    # add_constraints_to_armature(arm)

    return arm
