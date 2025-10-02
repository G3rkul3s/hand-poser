import bpy
import numpy as np
# import os
from pathlib import Path
from mathutils import Matrix, Vector, Euler
from math import radians

# === CONFIG ===
BONE_NAMES = [              # 16 MANO bones:
    "Wrist",                # "wrist",       0
    "IndexProximal",        # "index1",      1
    "IndexIntermadiate",    # "index2",      2
    "IndexDistal",          # "index3",      3
    "MiddleProximal",       # "middle1",     4
    "MiddleIntermadiate",   # "middle2",     5
    "MiddleDistal",         # "middle3",     6
    "LittleProximal",       # "pinky1",      7
    "LittleIntermadiate",   # "pinky2",      8
    "LittleDistal",         # "pinky3",      9
    "RingProximal",         # "ring1",      10
    "RingIntermadiate",     # "ring2",      11
    "RingDistal",           # "ring3",      12
    "ThumbMetacarpal",      # "thumb1",     13
    "ThumbProximal",        # "thumb2",     14
    "ThumbDistal",          # "thumb3",     15
    # NOTE: New bones should be appended at the end of the list
]
FINGERTIP_NAMES = [
    "ThumbTip",
    "IndexTip",
    "MiddleTip",
    "RingTip",
    "LittleTip",
]
FINGERTIPS = { # Vertex index:
    FINGERTIP_NAMES[0] : 745, # Thumb
    FINGERTIP_NAMES[1] : 333, # Point
    FINGERTIP_NAMES[2] : 444, # Middle
    FINGERTIP_NAMES[3] : 555, # Ring
    FINGERTIP_NAMES[4] : 672, # Pinky
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
    basis_path = ROOT_DIR / 'data' / f"basis_{hand}.txt"
    anatomical_consistent_basis = np.loadtxt(basis_path, dtype=float)
    data = np.load(mano_path)
    v_template = data['v_template']   # [778, 3]
    shapedirs = data['shapedirs']     # [10, 778, 3]
    faces = data['f']                 # [1538, 3]
    joints = data['J']
    weights = data['weights']
    return v_template, shapedirs, faces, joints, weights, anatomical_consistent_basis

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

def create_joint_armature(mesh, hand, joint_positions, bone_names, bone_parents, basis):
    armature_data = bpy.data.armatures.new(f"MANO_{hand}_Hand")
    armature_obj = bpy.data.objects.new(f"MANO_{hand}_Hand", armature_data)
    bpy.context.collection.objects.link(armature_obj)

    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='EDIT')
    bones = {}

    for i, name in enumerate(bone_names):
        bone = armature_data.edit_bones.new(name)
        head = joint_positions[i]
        x_axis = Vector(basis[i, 6:9]) * -1 # makes the system right handed
        y_axis = Vector(basis[i, 3:6])
        z_axis = Vector(basis[i, 0:3])
        bone.head = head
        rot = Matrix((x_axis, y_axis, z_axis)).transposed()
        bone.matrix = Matrix.Translation(bone.head) @ rot.to_4x4()
        bone.length = 0.04
        bones[name] = bone

    # add joints for fingertips
    for name, vert_index in FINGERTIPS.items():
        bone = armature_data.edit_bones.new(name)
        head = mesh.data.vertices[vert_index].co

        bone.head = head
        tail = head + Vector((0.0, 1.0, 0.0)) * 0.04
        bone.tail = tail
        finger = name.rpartition('Tip')[0]
        bone.align_orientation(bones[finger + "Distal"])
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
    constraint = root.constraints.new('LIMIT_ROTATION')
    constraint.use_limit_x = False
    constraint.use_limit_y = False
    constraint.use_limit_z = False
    constraint.owner_space = 'LOCAL'
    constraint.use_transform_limit = True
    for bone in root.children:
        if bone.name == BONE_NAMES[13]:     # if thumb
            # Proximal
            constraint = bone.constraints.new('LIMIT_ROTATION')
            constraint.min_x = radians(-90.0)
            constraint.max_x = radians(33.0)
            constraint.use_limit_x = True
            constraint.min_y = radians(-25.0)
            constraint.max_y = radians(25.0)
            constraint.use_limit_y = True
            constraint.use_limit_z = True
            constraint.owner_space = 'LOCAL'
            constraint.use_transform_limit = True
            # Intermadiate
            bone_1 = bone.children[0]
            constraint = bone_1.constraints.new('LIMIT_ROTATION')
            constraint.min_x = radians(-80.0)
            constraint.max_x = radians(0.0)
            constraint.use_limit_x = True
            constraint.use_limit_y = True
            constraint.use_limit_z = True
            constraint.owner_space = 'LOCAL'
            constraint.use_transform_limit = True
            # Distal
            bone_2 = bone_1.children[0]
            constraint = bone_2.constraints.new('LIMIT_ROTATION')
            constraint.min_x = radians(-90.0)
            constraint.max_x = radians(20.0)
            constraint.use_limit_x = True
            constraint.use_limit_y = True
            constraint.use_limit_z = True
            constraint.owner_space = 'LOCAL'
            constraint.use_transform_limit = True
            # Tip
            bone_3 = bone_2.children[0]
            constraint = bone_3.constraints.new('LIMIT_ROTATION')
            constraint.use_limit_x = True
            constraint.use_limit_y = True
            constraint.use_limit_z = True
            constraint.owner_space = 'LOCAL'
            constraint.use_transform_limit = True

        else:
            # Proximal
            constraint = bone.constraints.new('LIMIT_ROTATION')
            constraint.min_x = radians(-90.0)
            constraint.max_x = radians(15.0)
            constraint.use_limit_x = True
            if bone.name == BONE_NAMES[7]:      # if pinky
                constraint.min_y = 0.0
                constraint.max_y = 0.0
            elif bone.name == BONE_NAMES[10]:   # if ring
                constraint.min_y = radians(-5.0)
                constraint.max_y = radians(15.0)
            elif bone.name == BONE_NAMES[4]:    # if middle
                constraint.min_y = radians(-5.0)
                constraint.max_y = radians(15.0)
            elif bone.name == BONE_NAMES[1]:    # if index
                constraint.min_y = radians(-20.0)
                constraint.max_y = radians(15.0)
            constraint.use_limit_y = True
            constraint.use_limit_z = True
            constraint.owner_space = 'LOCAL'
            constraint.use_transform_limit = True
            # Intermadiate
            bone_1 = bone.children[0]
            constraint = bone_1.constraints.new('LIMIT_ROTATION')
            constraint.use_limit_x = True
            constraint.min_x = radians(-100.0)
            constraint.max_x = radians(5.0)
            constraint.use_limit_y = True
            constraint.use_limit_z = True
            constraint.owner_space = 'LOCAL'
            constraint.use_transform_limit = True
            # Distal
            bone_2 = bone_1.children[0]
            constraint = bone_2.constraints.new('LIMIT_ROTATION')
            constraint.use_limit_x = True
            constraint.min_x = radians(-90.0)
            constraint.max_x = radians(6.0)
            constraint.use_limit_y = True
            constraint.use_limit_z = True
            constraint.owner_space = 'LOCAL'
            constraint.use_transform_limit = True
            # Tip
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

    v_template, shapedirs, faces, joints, weights, basis = load_mano_model(hand)

    # === Create base mesh ===
    mesh = create_mano_mesh(f"MANO_{hand}_Hand_mesh", v_template, faces)

    # === Add shape keys ===
    add_shape_keys(mesh, shapedirs, v_template)
    
    # === Add armature ===
    arm = create_joint_armature(mesh, hand, joints, BONE_NAMES, BONE_PARENTS, basis)
    mesh.parent = arm
   
    # Set the origin of the armature to wrist bone
    # NOTE: this breaks the joint position update
    '''
    current_mode = bpy.context.mode
    bpy.ops.object.mode_set(mode='EDIT')
    root = arm.data.edit_bones.get(BONE_NAMES[0])
    bone_loc = root.head
    bpy.ops.object.mode_set(mode='OBJECT')
    cursor_loc = bpy.context.scene.cursor.location.copy()
    bpy.context.scene.cursor.location = bone_loc
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.origin_set(type='ORIGIN_CURSOR')
    bpy.context.scene.cursor.location = cursor_loc
    bpy.ops.view3d.snap_selected_to_cursor()
    arm.select_set(False)
    bpy.ops.object.mode_set(mode=current_mode)
    '''

    arm_mod = mesh.modifiers.new(name="ArmatureDeform", type='ARMATURE')
    arm_mod.object = arm
    assign_skinning_weights(mesh, weights, BONE_NAMES)

    add_constraints_to_armature(arm)


    return arm
