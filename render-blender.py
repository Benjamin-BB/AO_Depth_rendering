import argparse, sys, os, math, re
from typing import Tuple

import bpy, bmesh
from glob import glob
import random

## blender --background --python render_blender.py -- --output_folder /tmp path_to_model.obj ##
#test

parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
parser.add_argument('--views', type=int, default=30,
                    help='number of views to be rendered')
parser.add_argument('obj', type=str,
                    help='Path to the folder with obj files to be rendered.')
parser.add_argument('--output_folder', type=str, default='/tmp',
                    help='The path the output will be dumped to.')
parser.add_argument('--scale', type=float, default=1,
                    help='Scaling factor applied to model. Depends on size of mesh.')
parser.add_argument('--remove_doubles', type=bool, default=True,
                    help='Remove double vertices to improve mesh quality.')
parser.add_argument('--edge_split', type=bool, default=True,
                    help='Adds edge split filter.')
parser.add_argument('--depth_scale', type=float, default=1.4,
                    help='Scaling that is applied to depth. Depends on size of mesh. Try out various values until you get a good result. Ignored if format is OPEN_EXR.')
parser.add_argument('--color_depth', type=str, default='8',
                    help='Number of bit per channel used for output. Either 8 or 16.')
parser.add_argument('--format', type=str, default='PNG',
                    help='Format of files generated. Either PNG or OPEN_EXR')
parser.add_argument('--resolution', type=int, default=600,
                    help='Resolution of the images.')
parser.add_argument('--engine', type=str, default='BLENDER_EEVEE',
                    help='Blender internal engine for rendering. E.g. CYCLES, BLENDER_EEVEE, ...')
parser.add_argument('--random', type=bool, default=False,
                    help='Randomize selected objects')
parser.add_argument('--num', type=int, default=100,
                    help='number of object to render, to use only with random')
parser.add_argument('--job_id', type=int, default=1,
                    help='gives job id')
parser.add_argument('--num_job', type=int, default=1,
                    help='number of jobs')
parser.add_argument('--sample', type=int, default=512,
                    help='number of samples for blender')
parser.add_argument("--gltf", action="store_true")
parser.add_argument("--randmat", action="store_true")
parser.add_argument("--randlight", action="store_true")
parser.add_argument("--randcamera", action="store_true")
parser.add_argument("--randscale", action="store_true")
parser.add_argument("--denoise", action="store_true")
parser.add_argument("--cache", type=str, default="")
parser.add_argument("--hdri", type=str, default="")


argv = sys.argv[sys.argv.index("--") + 1:]
args = parser.parse_args(argv)

# <editor-fold desc="Description">
# Set up rendering
context = bpy.context
scene = bpy.context.scene
render = bpy.context.scene.render

bpy.context.preferences.addons[
    "cycles"
].preferences.compute_device_type = "CUDA" # or "OPENCL"
render.engine = args.engine
scene.cycles.device = 'GPU'

bpy.context.preferences.addons["cycles"].preferences.get_devices()
print(bpy.context.preferences.addons["cycles"].preferences.compute_device_type)
for d in bpy.context.preferences.addons["cycles"].preferences.devices:
    d["use"] = 1 # Using all devices, include GPU and CPU
    print(d["name"], d["use"])

render.image_settings.color_mode = 'RGBA' # ('RGB', 'RGBA', ...)
render.image_settings.color_depth = args.color_depth # ('8', '16')
render.image_settings.file_format = args.format # ('PNG', 'OPEN_EXR', 'JPEG, ...)
render.resolution_x = args.resolution
render.resolution_y = args.resolution
render.resolution_percentage = 100
render.film_transparent = False # True if we want to hide the background
scene.cycles.samples = args.sample
if args.denoise:
    bpy.context.scene.cycles.use_denoising = True


#test commit
scene.use_nodes = True
scene.view_layers["ViewLayer"].use_pass_normal = True
scene.view_layers["ViewLayer"].use_pass_diffuse_color = True
scene.view_layers["ViewLayer"].use_pass_object_index = True
scene.view_layers["ViewLayer"].use_pass_z = True
scene.view_layers["ViewLayer"].use_pass_ambient_occlusion = True
scene.view_layers["ViewLayer"].use_ao = True

print("View layers: ", scene.view_layers.keys())

nodes = bpy.context.scene.node_tree.nodes
links = bpy.context.scene.node_tree.links

# Clear default nodes
for n in nodes:
    nodes.remove(n)

# Create input render layer node
render_layers = nodes.new('CompositorNodeRLayers')

# Create AO output
ao_file_output = nodes.new(type="CompositorNodeOutputFile")
ao_file_output.label = 'AO Output'
ao_file_output.base_path = ''
ao_file_output.file_slots[0].use_node_format = True
ao_file_output.format.file_format = args.format
ao_file_output.format.color_mode = 'RGBA' #hdr a l'air d'avoir besoin de 32 bits ou d'un float, peut etre besoin de changer en RGB
ao_file_output.format.color_depth = args.color_depth
links.new(render_layers.outputs['AO'], ao_file_output.inputs[0])

# Create depth output nodes
depth_file_output = nodes.new(type="CompositorNodeOutputFile")
depth_file_output.label = 'Depth Output'
depth_file_output.base_path = ''
depth_file_output.file_slots[0].use_node_format = True
depth_file_output.format.file_format = args.format
depth_file_output.format.color_depth = args.color_depth
if args.format == 'OPEN_EXR':
    links.new(render_layers.outputs['Depth'], depth_file_output.inputs[0])
else:
    depth_file_output.format.color_mode = "BW"

    # Remap as other types can not represent the full range of depth.
    map = nodes.new(type="CompositorNodeMapValue")
    # Size is chosen kind of arbitrarily, try out until you're satisfied with resulting depth map.
    map.offset = [-0.7]
    map.size = [args.depth_scale]
    map.use_min = True
    map.min = [0]

    links.new(render_layers.outputs['Depth'], map.inputs[0])
    links.new(map.outputs[0], depth_file_output.inputs[0])

# Create normal output nodes
scale_node = nodes.new(type="CompositorNodeMixRGB")
scale_node.blend_type = 'MULTIPLY'
# scale_node.use_alpha = True
scale_node.inputs[2].default_value = (0.5, 0.5, 0.5, 1)
links.new(render_layers.outputs['Normal'], scale_node.inputs[1])

bias_node = nodes.new(type="CompositorNodeMixRGB")
bias_node.blend_type = 'ADD'
# bias_node.use_alpha = True
bias_node.inputs[2].default_value = (0.5, 0.5, 0.5, 0)
links.new(scale_node.outputs[0], bias_node.inputs[1])

normal_file_output = nodes.new(type="CompositorNodeOutputFile")
normal_file_output.label = 'Normal Output'
normal_file_output.base_path = ''
normal_file_output.file_slots[0].use_node_format = True
normal_file_output.format.file_format = args.format
links.new(bias_node.outputs[0], normal_file_output.inputs[0])

# Create albedo output nodes
#alpha_albedo = nodes.new(type="CompositorNodeSetAlpha")
#links.new(render_layers.outputs['DiffCol'], alpha_albedo.inputs['Image'])
#links.new(render_layers.outputs['Alpha'], alpha_albedo.inputs['Alpha'])

# albedo_file_output = nodes.new(type="CompositorNodeOutputFile")
# albedo_file_output.label = 'Albedo Output'
# albedo_file_output.base_path = ''
# albedo_file_output.file_slots[0].use_node_format = True
# albedo_file_output.format.file_format = args.format
# albedo_file_output.format.color_mode = 'RGBA'
# albedo_file_output.format.color_depth = args.color_depth
# links.new(alpha_albedo.outputs['Image'], albedo_file_output.inputs[0])

# Create id map output nodes
id_file_output = nodes.new(type="CompositorNodeOutputFile")
id_file_output.label = 'ID Output'
id_file_output.base_path = ''
id_file_output.file_slots[0].use_node_format = True
id_file_output.format.file_format = args.format
id_file_output.format.color_depth = args.color_depth

if args.format == 'OPEN_EXR':
    links.new(render_layers.outputs['IndexOB'], id_file_output.inputs[0])
else:
    id_file_output.format.color_mode = 'BW'
    links.new(render_layers.outputs['IndexOB'], id_file_output.inputs[0])
    # divide_node = nodes.new(type='CompositorNodeMath')
    # divide_node.operation = 'DIVIDE'
    # divide_node.use_clamp = False
    # divide_node.inputs[1].default_value = 2**int(args.color_depth)

    # links.new(render_layers.outputs['IndexOB'], divide_node.inputs[0])
    # links.new(divide_node.outputs[0], id_file_output.inputs[0])

# Delete default cube
try:
    context.active_object.select_set(True)
    bpy.ops.object.delete()
except:
    print('No object in the scene')

def create_texture_node(node_tree: bpy.types.NodeTree, path: str, is_color_data: bool) -> bpy.types.Node:
    # Instantiate a new texture image node
    texture_node = node_tree.nodes.new(type='ShaderNodeTexImage')
    # Open an image and set it to the node
    texture_node.image = bpy.data.images.load(path)
    # Set other parameters
    texture_node.image.colorspace_settings.is_data = False if is_color_data else True
    # Return the node
    return texture_node

def set_principled_node(principled_node: bpy.types.Node,
                        base_color: Tuple[float, float, float, float] = (0.6, 0.6, 0.6, 1.0),
                        subsurface: float = 0.0,
                        subsurface_color: Tuple[float, float, float, float] = (0.8, 0.8, 0.8, 1.0),
                        subsurface_radius: Tuple[float, float, float] = (1.0, 0.2, 0.1),
                        metallic: float = 0.0,
                        specular: float = 0.5,
                        specular_tint: float = 0.0,
                        roughness: float = 0.5,
                        anisotropic: float = 0.0,
                        anisotropic_rotation: float = 0.0,
                        sheen: float = 0.0,
                        sheen_tint: float = 0.5,
                        clearcoat: float = 0.0,
                        clearcoat_roughness: float = 0.03,
                        ior: float = 1.45,
                        transmission: float = 0.0,
                        transmission_roughness: float = 0.0,
                        alpha: float = 1.0) -> None:
    principled_node.inputs['Base Color'].default_value = base_color
    principled_node.inputs['Subsurface'].default_value = subsurface
    principled_node.inputs['Subsurface Color'].default_value = subsurface_color
    principled_node.inputs['Subsurface Radius'].default_value = subsurface_radius
    principled_node.inputs['Metallic'].default_value = metallic
    principled_node.inputs['Specular'].default_value = specular
    principled_node.inputs['Specular Tint'].default_value = specular_tint
    principled_node.inputs['Roughness'].default_value = roughness
    principled_node.inputs['Anisotropic'].default_value = anisotropic
    principled_node.inputs['Anisotropic Rotation'].default_value = anisotropic_rotation
    principled_node.inputs['Sheen'].default_value = sheen
    principled_node.inputs['Sheen Tint'].default_value = sheen_tint
    principled_node.inputs['Clearcoat'].default_value = clearcoat
    principled_node.inputs['Clearcoat Roughness'].default_value = clearcoat_roughness
    principled_node.inputs['IOR'].default_value = ior
    principled_node.inputs['Transmission'].default_value = transmission
    principled_node.inputs['Transmission Roughness'].default_value = transmission_roughness
    principled_node.inputs['Alpha'].default_value = alpha

def random_material(mat):
    set_principled_node(mat.node_tree.nodes["Principled BSDF"], 
        base_color=(random.random(),random.random(),random.random(), 1.0),
        metallic=random.random(),
        roughness=random.random()
        )

def create_plane(scale=1.0):
    """Create plane geometry"""
    mesh = bpy.data.meshes.new("Plane")
    obj = bpy.data.objects.new("Plane", mesh)

    bpy.context.collection.objects.link(obj)

    bm = bmesh.new()
    bm.from_object(obj, bpy.context.view_layer.depsgraph)

    s = scale
    bm.verts.new((s,s,0))
    bm.verts.new((s,-s,0))
    bm.verts.new((-s,s,0))
    bm.verts.new((-s,-s,0))

    bmesh.ops.contextual_create(bm, geom=bm.verts)

    bm.to_mesh(mesh)
    return obj

# Import textured mesh
bpy.ops.object.select_all(action='DESELECT')
# </editor-fold>

directory = args.obj
j = 0

random.seed(args.job_id) 

# Look at all HDRI maps
hdri_files = []
hdri_node = None
hdri_mapping_node = None
if args.hdri != "":
    world =  bpy.data.worlds['World']
    world.use_nodes = True
    hdri_files = glob(args.hdri + '/**/*.exr', recursive=True)
    
    # Create HDRI node (for env map)
    hdri_node = world.node_tree.nodes.new(type="ShaderNodeTexEnvironment")
    back_node = world.node_tree.nodes['World Output']
    world.node_tree.links.new(hdri_node.outputs['Color'], back_node.inputs['Surface'])
    # Create Mapping node (to generate transformation)
    hdri_mapping_node = world.node_tree.nodes.new(type="ShaderNodeMapping")
    world.node_tree.links.new(hdri_mapping_node.outputs["Vector"], hdri_node.inputs["Vector"])
    # Create input texture coordinates
    hdri_texcoords_node = world.node_tree.nodes.new(type="ShaderNodeTexCoord")
    world.node_tree.links.new(hdri_texcoords_node.outputs["Generated"], hdri_mapping_node.inputs["Vector"])

    # Final link and clean the default light
    bpy.ops.object.delete({"selected_objects": [bpy.data.lights['Light']]})
else:
    # Add another light
    bpy.ops.object.light_add(type='SUN')


if args.cache != "":
    list_objects = open(args.cache, "r").readlines()
    list_objects = [v.strip() for v in list_objects]
else:
    if args.gltf:
        list_objects = glob(directory + '/**/*.gltf', recursive=True)
    else:
        list_objects = glob(directory + '/**/*.obj', recursive=True)

if args.random == True:
    N = args.num
    list_objects = random.sample(list_objects, N)

part = len(list_objects)//args.num_job
j = (args.job_id-1) * part

bpy.context.scene.render.use_persistent_data = False

for p in range((args.job_id-1)*part, (args.job_id)*part):
    name = list_objects[p]
    w = 0
    
    # Delete all objects
    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.delete()

    # Add plane
    plane = create_plane(scale=1000)
    plane.is_shadow_catcher = True
    if args.randmat:
        plane_material = bpy.data.materials.new(name="Plane BSDF")
        plane_material.use_nodes = True
        random_material(plane_material)
        plane.data.materials.append(plane_material)
    
    # Render object
    j = j+1
    if args.gltf:
        print(f"Load: {name}")
        bpy.ops.import_scene.gltf(filepath=name, merge_vertices=True)
    else:
        bpy.ops.import_scene.obj(filepath=name)
        obj = bpy.context.selected_objects[0]
        context.view_layer.objects.active = obj
        # Apply correction only for obj objects
        if args.remove_doubles:
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.remove_doubles()
            bpy.ops.object.mode_set(mode='OBJECT')
        if args.edge_split:
            bpy.ops.object.modifier_add(type='EDGE_SPLIT')
            context.object.modifiers["EdgeSplit"].split_angle = 1.32645
            bpy.ops.object.modifier_apply(modifier="EdgeSplit")

    print(f"Number objects: {len(bpy.context.selected_objects)}")
    w = 10000.0
    if args.randscale:
        scale = 0.1 + random.random()*10.0
    else:
        scale = args.scale
    for obj in bpy.context.selected_objects[:1]:  
        print(f" - {obj.name}")
        w = min(obj.bound_box[0][1]*scale, w)
   
    # Translation de l'objet sur le plan (z=0) and scale
    obj = bpy.context.selected_objects[0]
    obj.pass_index = 1
    context.view_layer.objects.active = obj
    obj.location = [0, 0, -w]
    if args.scale != 1:
        bpy.ops.transform.resize(value=(args.scale,args.scale,args.scale))
        bpy.ops.object.transform_apply(scale=True)
    
    # Make light just directional, disable shadows.
    if hdri_node:
        # Change image
        image = random.sample(hdri_files, 1)[0]
        if hdri_node.image != None:
            hdri_node.image.user_clear()
            bpy.data.images.remove(hdri_node.image)
        print(f"Image: {image}")
        hdri_node.image = bpy.data.images.load(image)
        hdri_node.image.alpha_mode = 'NONE'
        # Change rotation
        hdri_mapping_node.inputs["Rotation"].default_value[2] = math.radians(360*random.random())
    else:
        light = bpy.data.lights['Light']
        light.type = 'SUN'
        light.use_shadow = True
        light.energy = 10.0

        light2 = bpy.data.lights['Sun']
        light2.use_shadow = True
        light2.energy = 0.015
        bpy.data.objects['Sun'].rotation_euler = bpy.data.objects['Light'].rotation_euler
        bpy.data.objects['Sun'].rotation_euler[0] += math.radians(180)
        bpy.data.objects['Sun'].rotation_mode = 'ZYX'
        bpy.data.objects['Light'].rotation_mode = 'ZYX'

    # Place camera
    cam = scene.objects['Camera']
    cam.location = (0, 1, 0.6)
    cam.data.lens = 35
    cam.data.sensor_width = 32
    cam.data.clip_end = 1000
    cam.data.clip_start = 0.0001

    cam_constraint = cam.constraints.new(type='TRACK_TO')
    cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    cam_constraint.up_axis = 'UP_Y'

    cam_empty = bpy.data.objects.new("Empty", None)
    cam_empty.location = (0, 0, 0)
    cam.parent = cam_empty
    cam_empty.rotation_mode = 'XYZ' # See why this is the correct mode for the camera

    scene.collection.objects.link(cam_empty)
    context.view_layer.objects.active = cam_empty
    cam_constraint.target = cam_empty

    stepsize = 360.0 / args.views

    fp = os.path.abspath(args.output_folder) + os.path.sep

    for i in range(0, args.views):
        if(args.randmat):
            random_material(plane_material)
            for slot in obj.material_slots:
                random_material(slot.material)

        if hdri_node:
            if(args.randlight):
                # Change texture
                # TODO: Evaluate if it does not have too much performance penality on the cluster
                image = random.sample(hdri_files, 1)[0]
                if hdri_node.image != None:
                    hdri_node.image.user_clear()
                    bpy.data.images.remove(hdri_node.image)
                hdri_node.image = bpy.data.images.load(image)
                hdri_node.image.alpha_mode = 'NONE'
                # Change rotation
                hdri_mapping_node.inputs["Rotation"].default_value[2] = math.radians(360*random.random())
        else:
            if(args.randlight):
                bpy.data.objects['Sun'].rotation_euler[0] = math.radians(85*random.random())
                bpy.data.objects['Sun'].rotation_euler[2] = math.radians(360*random.random())
                bpy.data.objects['Light'].rotation_euler[0] = math.radians(85*random.random())
                bpy.data.objects['Light'].rotation_euler[2] = math.radians(360*random.random())

        if(args.randcamera):
            cam_empty.rotation_euler[2] = math.radians(random.random()*360)
            cam_empty.rotation_euler[0] = math.radians(random.random()*80)
            dist = 1.7 * random.random() + 0.5
            cam.location = (0, 1*dist*scale, 0.0)
            cam.data.lens = 20 + random.random()*20
            cam.data.sensor_width = 20 + random.random()*20

        print("Rotation {}, {}".format((stepsize * i), math.radians(stepsize * i)))

        render_file_path = fp + str(j) +'_r_{0:03d}'.format(int(i))
        print(f"Output: {render_file_path}")
        scene.render.filepath = render_file_path
        depth_file_output.file_slots[0].path = render_file_path + "_depth"
        ao_file_output.file_slots[0].path = render_file_path + "_ao"
        id_file_output.file_slots[0].path = render_file_path + "_id"
        normal_file_output.file_slots[0].path = render_file_path + "_normal"
        print("Render layers", render_layers.outputs.keys())


        bpy.ops.render.render(write_still=True)  # render still
        bpy.context.scene.render.use_persistent_data = True
        cam_empty.rotation_euler[2] += math.radians(stepsize)
        