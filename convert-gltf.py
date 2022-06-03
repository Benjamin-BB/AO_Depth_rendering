import glob
import argparse
import subprocess

parser = argparse.ArgumentParser(description='Convert all obj to gltf via assimp')
parser.add_argument('directory', type=str,
                    help='Path to the folder with obj files to be rendered.')
args = parser.parse_args()

list_objects = glob.glob(args.directory + '/**/*.obj', recursive=True)
for ob in list_objects:
    name_without_ext = ".".join(ob.split(".")[:-1])
    print(f" - {ob} -> {name_without_ext}.gltf")
    list_files = subprocess.run(["assimp", "export", ob, name_without_ext+".gltf","-f","gltf"])
