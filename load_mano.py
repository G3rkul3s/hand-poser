import bpy
import numpy as np
# import os
from pathlib import Path
from mathutils import Vector

# === CONFIG ===
MANO_BONE_NAMES = [
    "wrist",
    "index1",
    "index2",
    "index3",
    "middle1",
    "middle2",
    "middle3",
    "pinky1",
    "pinky2",
    "pinky3",
    "ring1",
    "ring2",
    "ring3",
    "thumb1",
    "thumb2",
    "thumb3",
]
FINGERTIPS = {
    "thumb4"  : 744,
    "index4"  : 320,
    "middle4" : 443,
    "ring4"   : 554,
    "pinky4"  : 671,
}
BONE_PARENTS = {
    "wrist"   : None,
    "thumb1"  : "wrist",
    "thumb2"  : "thumb1",
    "thumb3"  : "thumb2",
    "thumb4"  : "thumb3", # Right/Left - 744
    "index1"  : "wrist",
    "index2"  : "index1",
    "index3"  : "index2",
    "index4"  : "index3", # Right/Left - 320
    "middle1" : "wrist",
    "middle2" : "middle1",
    "middle3" : "middle2",
    "middle4" : "middle3", # Right/Left - 443
    "ring1"   : "wrist",
    "ring2"   : "ring1",
    "ring3"   : "ring2",
    "ring4"   : "ring3", # Right/Left - 554
    "pinky1"  : "wrist",
    "pinky2"  : "pinky1",
    "pinky3"  : "pinky2",
    "pinky4"  : "pinky3", # Right/Left - 671
}

def load_mano_model(hand):
    ROOT_DIR = Path(__file__).parent
    mano_path = ROOT_DIR / 'data' / f"MANO_{hand}.npz"
    data = np.load(mano_path)
    v_template = data['v_template']          # [778, 3]
    shapedirs = data['shapedirs']            # [10, 778, 3]
    faces = data['f']                    # [1538, 3]
    joints = data['J']
    weights = data['weights']
    return v_template, shapedirs, faces, joints, weights

def load_regressor(hand):
    ROOT_DIR = Path(__file__).parent
    mano_path = ROOT_DIR / 'data' / f"MANO_{hand}.npz"
    data = np.load(mano_path)
    j_regressor = data['J_regressor']
    joints = data['J']
    return j_regressor, joints

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
        tail = head + (0.0, 0.04, 0.0)  # small length bone
        bone.head = head
        bone.tail = tail
        bones[name] = bone

    # add joints for fingertips
    for name, index in FINGERTIPS.items():
        bone = armature_data.edit_bones.new(name)
        head = mesh.data.vertices[index].co
        tail = head + Vector((0.0, 0.04, 0.0))  # small length bone
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
    arm = create_joint_armature(mesh, hand, joints, MANO_BONE_NAMES, BONE_PARENTS)
    mesh.parent = arm
    
    arm_mod = mesh.modifiers.new(name="ArmatureDeform", type='ARMATURE')
    arm_mod.object = arm
    assign_skinning_weights(mesh, weights, MANO_BONE_NAMES)

    return arm
