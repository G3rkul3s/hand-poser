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
BONE_TAILS = [
    # Vector((0., 0., 1.)),
    # Vector((-0.01867034, -0.11700923,  0.9929553 )),
    # Vector(( 0.10213755, -0.1443118,   0.98424697)),
    # Vector(( 0.03692146, -0.03023705,  0.99886066)),
    # Vector((-0.07896362,  0.09917654,  0.99193186)),
    # Vector((-0.2199986,   0.1445599,   0.96472955)),
    # Vector((-0.22161944,  0.2364873,   0.9460225 )),
    # Vector((0.04056842, 0.46810475, 0.8827413 )),
    # Vector((-0.08559407,  0.61912215,  0.78061604)),
    # Vector((-0.39246565,  0.70662427,  0.58877224)),
    # Vector((0.03632812, 0.15820135, 0.9867384 )),
    # Vector((-0.17324242,  0.3642259,   0.9150556 )),
    # Vector((-0.32323194,  0.48535645,  0.8123733 )),
    # Vector((-0.20466185, -0.75259537,  0.62587035)),
    # Vector((-0.2826093,  -0.34437987,  0.89528453)),
    # Vector((-0.20847481, -0.77505016,  0.59651923)),
    Vector((0.0, 1.0, 0.0)),
    Vector((-0.40589133,  0.9084971,   0.09942483)),
    Vector((-0.9175034,   0.36865664,  0.14926444)),
    Vector((-0.9794051,   0.19745013,  0.04217946)),
    Vector((-0.50082207,  0.8564043,  -0,.12549445)),
    Vector((-0.91782343,  0.3043403,  -0,.25490594)),
    Vector((-0.9747318,  -0.02589237, -0,.22187243)),
    Vector((-0.6034986,   0.7155929,  -0,.35173327)),
    Vector((-0.91077906,  0.2690159,  -0,.3132283 )),
    Vector((-0.8515483,  -0.0372237,  -0,.52295315)),
    Vector((-0.59668416,  0.79550135, -0,.10557305)),
    Vector((-0.9379732,   0.22229752, -0,.26606393)),
    Vector((-0.94602346, -0.18720707, -0,.26456177)),
    Vector((-0.18978268,  0.6577811,   0.72890776)),
    Vector((-0.01351721,  0.9346688,   0.35526258)),
    Vector((-0.41322485,  0.622609,    0.6645323 )),
]

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
        tail = head + Vector((0.0, 1.0, 0.0)) * 0.04 # BONE_TAILS[i].xyz
        bone.head = head
        bone.tail = tail
        bones[name] = bone

    # add joints for fingertips
    for name, index in FINGERTIPS.items():
        bone = armature_data.edit_bones.new(name)
        head = mesh.data.vertices[index].co
        # TODO: tail of the bones in the fingers should go through vertices
        tail = head + Vector((0.0, 1.0, 0.0)) * 0.04  # small length bone
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
    
    root = armature.pose.bones.get(MANO_BONE_NAMES[0])

    for bone in root.children:
        if bone.name == MANO_BONE_NAMES[13]:
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
            if bone.name == MANO_BONE_NAMES[7]: # if pinky
                constraint.max_y = 0.77
            elif bone.name == MANO_BONE_NAMES[10]: # if ring
                constraint.max_y = 0.38
            elif bone.name == MANO_BONE_NAMES[4]: # if middle
                constraint.max_y = 0.2
            elif bone.name == MANO_BONE_NAMES[1]: # if index
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
    arm = create_joint_armature(mesh, hand, joints, MANO_BONE_NAMES, BONE_PARENTS)
    mesh.parent = arm
    
    arm_mod = mesh.modifiers.new(name="ArmatureDeform", type='ARMATURE')
    arm_mod.object = arm
    assign_skinning_weights(mesh, weights, MANO_BONE_NAMES)

    # add_constraints_to_armature(arm)

    return arm
