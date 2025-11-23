import bpy

# Load the addon
# addon_path = ".../handposer_blender_addon.zip"
# # sys.path.append(addon_path)
# bpy.ops.preferences.addon_install(
#     'EXEC_DEFAULT',
#     filepath=addon_path,
#     overwrite=True)

# Enable the addon
addon_name = "handposer_blender_addon"
if addon_name not in bpy.context.preferences.addons:
    bpy.ops.preferences.addon_enable(module=addon_name)

scene = bpy.context.scene
# Select the save folder
scene.save_folder = "..."
# Export metadata
bpy.ops.view3d.export_metadata('EXEC_DEFAULT')
# Render animation
bpy.ops.view3d.multiview_render('EXEC_DEFAULT')
